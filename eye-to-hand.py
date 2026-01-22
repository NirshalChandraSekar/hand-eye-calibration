import os
import glob
import time
import numpy as np
import cv2

from zeus_robot import zeusRobot
from realsense import RealSenseCamera


# ---------------- Utilities ----------------
def invert_transform(R, t):
    """Invert rigid transform: x_dst = R x_src + t  ->  x_src = R^T x_dst + (-R^T t)."""
    R_inv = R.T
    t_inv = -R_inv @ t
    return R_inv, t_inv


def ur_pose6_to_Rt(pose6):
    """
    UR pose: [x, y, z, rx, ry, rz] where (rx,ry,rz) is Rodrigues rotation vector (axis-angle).
    Returns R (3x3), t (3x1) for gripper->base.
    """
    pose6 = np.asarray(pose6, dtype=np.float64).reshape(-1)
    if pose6.size != 6:
        raise ValueError(f"Expected 6 values, got {pose6.size}")
    t = pose6[:3].reshape(3, 1)
    rvec = pose6[3:].reshape(3, 1)
    R, _ = cv2.Rodrigues(rvec)
    return R, t


def load_ur_pose6_txt(path):
    """Load np.savetxt file containing 6 lines: x y z rx ry rz."""
    vals = np.loadtxt(path).astype(np.float64).reshape(-1)
    return ur_pose6_to_Rt(vals)


def Rt_to_T(R, t):
    """Make 4x4 homogeneous matrix."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T


def parse_pose_index(path):
    """
    Extract pose index from filename containing 'pose{N}'.
    Example: rgb_frame_f1380660_pose12.png -> 12
    """
    base = os.path.basename(path)
    try:
        return int(base.split("pose")[-1].split(".")[0])
    except Exception:
        raise ValueError(f"Could not parse pose index from: {path}")


def get_realsense_intrinsics_and_distortion(realsense, camera_serial=None):
    """
    Best-effort: try to fetch K and D from your RealSenseCamera wrapper.

    You MAY need to adapt this function to your RealSenseCamera implementation.
    Common patterns:
      - realsense.K / realsense.D
      - realsense.intrinsics / realsense.distortion_coeffs
      - realsense.get_intrinsics(serial) -> (K, D)
      - realsense.get_camera_params(...) -> dict with K,D

    Returns:
      K: (3,3) float64
      D: (N,1) float64
    """
    # 1) Direct attributes
    for k_attr in ["K", "intrinsics", "camera_matrix"]:
        if hasattr(realsense, k_attr):
            K = getattr(realsense, k_attr)
            if K is not None:
                K = np.asarray(K, dtype=np.float64).reshape(3, 3)
                break
    else:
        K = None

    for d_attr in ["D", "distortion_coeffs", "dist_coeffs"]:
        if hasattr(realsense, d_attr):
            D = getattr(realsense, d_attr)
            if D is not None:
                D = np.asarray(D, dtype=np.float64).reshape(-1, 1)
                break
    else:
        D = None

    if K is not None and D is not None:
        return K, D

    # 2) Methods
    for meth in ["get_intrinsics", "get_camera_params", "get_calibration", "get_params"]:
        if hasattr(realsense, meth):
            fn = getattr(realsense, meth)
            try:
                out = fn(camera_serial) if camera_serial is not None else fn()
            except TypeError:
                # maybe expects cameras=[serial]
                out = fn(cameras=[camera_serial]) if camera_serial is not None else fn()

            if isinstance(out, tuple) and len(out) >= 2:
                K = np.asarray(out[0], dtype=np.float64).reshape(3, 3)
                D = np.asarray(out[1], dtype=np.float64).reshape(-1, 1)
                return K, D

            if isinstance(out, dict):
                if "K" in out and "D" in out:
                    K = np.asarray(out["K"], dtype=np.float64).reshape(3, 3)
                    D = np.asarray(out["D"], dtype=np.float64).reshape(-1, 1)
                    return K, D

    raise RuntimeError(
        "Could not obtain camera intrinsics/distortion from RealSenseCamera.\n"
        "Fix by either:\n"
        "  (A) editing get_realsense_intrinsics_and_distortion() to match your wrapper, or\n"
        "  (B) hard-coding K and D in __init__ of eyeToHand.\n"
    )


def calibrate_eye_to_hand(R_gripper2base_list, t_gripper2base_list,
                          R_target2cam_list, t_target2cam_list):
    """
    Eye-to-hand: camera fixed, target mounted on gripper.
    OpenCV calibrateHandEye is naturally for eye-in-hand (solves for gripper_T_camera).
    For eye-to-hand, we invert robot poses to base->gripper before passing them in,
    so returned X corresponds to base_T_camera.
    """
    # Convert gripper->base to base->gripper
    R_base2gripper, t_base2gripper = [], []
    for R_g2b, t_g2b in zip(R_gripper2base_list, t_gripper2base_list):
        R_b2g, t_b2g = invert_transform(R_g2b, t_g2b)
        R_base2gripper.append(R_b2g)
        t_base2gripper.append(t_b2g)

    R_bc, t_bc = cv2.calibrateHandEye(
        R_gripper2base=R_base2gripper,
        t_gripper2base=t_base2gripper,
        R_target2cam=R_target2cam_list,
        t_target2cam=t_target2cam_list,
        # method=cv2.CALIB_HAND_EYE_TSAI  # optional: set method explicitly
    )
    return R_bc, t_bc


# ---------------- Eye-to-Hand Class ----------------
class eyeToHand:
    def __init__(self, zeus, realsense, which_robot="lightning",
                 tag_id=5, aruco_size_m=0.05,
                 intrinsics=None, distortion_coeffs=None):
        self.zeus = zeus
        self.realsense = realsense
        self.which_robot = which_robot

        self.tag_id = int(tag_id)
        self.aruco_size = float(aruco_size_m)

        # ArUco setup
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
        self.aruco_params = cv2.aruco.DetectorParameters()

        # Camera calibration
        self.K = None if intrinsics is None else np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
        self.D = None if distortion_coeffs is None else np.asarray(distortion_coeffs, dtype=np.float64).reshape(-1, 1)

    def capture_frames(self, cameras, save_path="frames", eef_save_dir="eef_poses"):
        if save_path is not None and not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(eef_save_dir):
            os.makedirs(eef_save_dir)

        input("Press enter when the marker is set in the gripper to grasp...")
        self.zeus.grasp(enable=True, robot=[self.which_robot])
        time.sleep(2)

        self.realsense.start_pipeline(cameras=cameras)

        robot_joint_states = [
            [-3.232577149068014, -2.8741971455016078, -0.8417272567749023, -2.274726530114645, 0.9309292435646057, -0.042792622243062794],
            [-3.2409077326404017, -2.939486642877096, -1.072386622428894, -2.1844521961607875, 0.7663727402687073, 0.005993842147290707],
            [-3.258474890385763, -2.4025961361327113, -1.4142531156539917, -1.749876161614889, 0.8947597146034241, -0.1420834700213831],
            [-3.1617606321917933, -2.411602636376852, -1.4207221269607544, -1.6822368107237757, 0.67801833152771, -0.15445167223085576],
            [-3.0208659807788294, -3.456479688683981, -0.49243199825286865, -2.3539487324156703, 0.6776517629623413, -0.2649191061602991],
            [-3.3034854570971888, -3.429298540154928, -0.28306570649147034, -2.4287530384459437, 0.49943268299102783, -0.26154882112611944],
            [-3.3035247961627405, -2.901008268395895, -0.8352807760238647, -2.1861101589598597, 0.6200353503227234, -0.08052045503725225],
            [-3.1145761648761194, -2.4509479008116664, -1.4737581014633179, -2.0544902286925257, 0.6199358701705933, 0.39794135093688965],
            [-3.0568314234363, -2.9231282673277796, -0.8685926198959351, -2.8223730526366175, 0.9313368201255798, 0.8624715805053711],
            [-3.2915309111224573, -3.2087956867613734, -0.8688321113586426, -1.8140560589232386, 0.9307109117507935, -0.6583541075335901],
            [-3.284423891698019, -3.259158273736471, -0.8751027584075928, -1.8181115589537562, 0.6487405300140381, -0.595102612172262],
            [-3.193688694630758, -3.066599508325094, -0.8549597859382629, -2.2734886608519496, 0.6486448645591736, -0.03683597246278936],
            [-3.3377564589129847, -3.066179414788717, -0.8559580445289612, -2.2853170833983363, 0.6487679481506348, -0.03683597246278936],
            [-3.152336899434225, -3.193012853662008, -0.8559027910232544, -2.288351675073141, 0.6488037109375, -0.03856212297548467],
            [-3.173217837010519, -2.250906606713766, -1.7732235193252563, -1.334277705555298, 0.47437041997909546, -0.37476951280702764],
            [-3.3571882883654993, -3.0163284740843714, -0.5756209492683411, -2.265193601647848, 0.945897102355957, -0.10583335558046514],
            [-3.2406166235553187, -3.1282993755736292, -0.5478165149688721, -2.3829032383360804, 0.9459531307220459, -0.10504085222353154],
            [-3.4826021830188196, -3.128998418847555, -0.5479365587234497, -2.383430620233053, 0.9459209442138672, -0.10504085222353154],
            [-3.4826138655291956, -2.939977308312887, -0.5469319224357605, -2.3834029636778773, 0.9459279179573059, -0.10504419008363897],
            [-3.3272910753833216, -3.383951803246969, -0.3934568166732788, -2.361539979974264, 0.9451899528503418, -0.10531360307802373]
        ]

        for idx, joint_state in enumerate(robot_joint_states, start=1):
            print(f"Moving to joint state {idx}/{len(robot_joint_states)}")
            self.zeus.move_to_joint_pose(lightning_joint_pose=joint_state)
            time.sleep(2)

            frame_filename = os.path.join(save_path, f"rgb_frame_{cameras[0]}_pose{idx}.png")
            self.realsense.save_rgb_frame(cameras=cameras, path=frame_filename)

            eef_poses = self.zeus.get_current_eef_poses()
            eef_pose = eef_poses[self.which_robot]  # expect [x,y,z,rx,ry,rz]
            np.savetxt(os.path.join(eef_save_dir, f"eef_pose_{self.which_robot}_pose{idx}.txt"),
                       np.array(eef_pose, dtype=np.float64))

        self.realsense.stop_pipeline()

    def _ensure_KD(self, camera_serial):
        if self.K is not None and self.D is not None:
            return
        self.K, self.D = get_realsense_intrinsics_and_distortion(self.realsense, camera_serial=camera_serial)

    def get_aruco_pose_target2cam(self, image):
        """
        Detect tag_id and return (R_target2cam, t_target2cam) such that:
            X_cam = R_target2cam * X_target + t_target2cam
        """
        corners, ids, _ = cv2.aruco.detectMarkers(image, self.aruco_dict, parameters=self.aruco_params)
        if ids is None:
            return None, None

        ids_flat = ids.flatten()
        if self.tag_id not in ids_flat:
            return None, None

        idx = int(np.where(ids_flat == self.tag_id)[0][0])

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, self.aruco_size, self.K, self.D
        )
        rvec = rvecs[idx].reshape(3, 1)
        tvec = tvecs[idx].reshape(3, 1)

        R, _ = cv2.Rodrigues(rvec)
        return R, tvec

    def calibrate(self, path_to_frames="frames", path_to_eef_poses="eef_poses",
                  camera_serial="f1380660", print_debug=True):
        """
        Reads saved images + UR pose6 txt files, runs eye-to-hand calibration, prints base_T_camera.
        """
        self._ensure_KD(camera_serial)

        frame_paths = sorted(
            glob.glob(os.path.join(path_to_frames, f"rgb_frame_{camera_serial}_pose*.png")),
            key=parse_pose_index
        )
        if len(frame_paths) == 0:
            raise FileNotFoundError(f"No frames found in {path_to_frames} for serial={camera_serial}")

        R_g2b_list, t_g2b_list = [], []
        R_t2c_list, t_t2c_list = [], []

        used = 0
        skipped = 0

        for fp in frame_paths:
            pose_i = parse_pose_index(fp)
            eef_path = os.path.join(path_to_eef_poses, f"eef_pose_{self.which_robot}_pose{pose_i}.txt")
            if not os.path.exists(eef_path):
                if print_debug:
                    print(f"[SKIP] missing eef pose: {eef_path}")
                skipped += 1
                continue

            img = cv2.imread(fp)
            if img is None:
                if print_debug:
                    print(f"[SKIP] could not read image: {fp}")
                skipped += 1
                continue

            R_t2c, t_t2c = self.get_aruco_pose_target2cam(img)
            if R_t2c is None:
                if print_debug:
                    print(f"[SKIP] ArUco not detected in pose {pose_i}: {fp}")
                skipped += 1
                continue

            R_g2b, t_g2b = load_ur_pose6_txt(eef_path)

            R_g2b_list.append(R_g2b)
            t_g2b_list.append(t_g2b)
            R_t2c_list.append(R_t2c)
            t_t2c_list.append(t_t2c)
            used += 1

        if used < 5:
            raise RuntimeError(f"Not enough valid samples for calibration. used={used}, skipped={skipped}")

        # --- Eye-to-hand calibration -> returns base_T_camera ---
        R_bc, t_bc = calibrate_eye_to_hand(R_g2b_list, t_g2b_list, R_t2c_list, t_t2c_list)
        T_bc = Rt_to_T(R_bc, t_bc)

        print("\n==================== RESULT ====================")
        print(f"Used samples: {used} (skipped {skipped})")
        print("R_base2camera:\n", R_bc)
        print("t_base2camera (m):\n", t_bc.reshape(3))
        print("\nbase_T_camera (4x4):\n", T_bc)
        print("================================================\n")

        return R_bc, t_bc, T_bc


# ---------------- Main ----------------
if __name__ == "__main__":
    # zeus = zeusRobot(robots=["lightning"])
    # zeus.go_home()
    # time.sleep(1)

    realsense = RealSenseCamera()

    # If your RealSenseCamera wrapper cannot provide intrinsics automatically,
    # pass K and D explicitly here.
    eye_to_hand = eyeToHand(
        None,
        realsense,
        which_robot="lightning",
        tag_id=5,
        aruco_size_m=0.05,
        intrinsics=None,          # or np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float64)
        distortion_coeffs=None    # or np.array([k1,k2,p1,p2,k3], dtype=np.float64)
    )

    cameras = ["f1380660"]

    # 1) Collect data (uncomment to capture new data)
    # eye_to_hand.capture_frames(cameras=cameras, save_path="frames", eef_save_dir="eef_poses")

    # 2) Calibrate from saved data and print base_T_camera
    eye_to_hand.calibrate(
        path_to_frames="frames",
        path_to_eef_poses="eef_poses",
        camera_serial=cameras[0],
        print_debug=True
    )
