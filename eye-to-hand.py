import pyrealsense2 as rs
import rtde_control
import rtde_receive
import numpy as np
import time
import cv2
import os

from zeus_robot import zeusRobot
from realsense import RealSenseCamera


# ---------------- Eye-to-Hand Class ----------------
class eyeToHand:
    def __init__(self, zeus, realsense, which_robot='lightning'):
        self.zeus = zeus
        self.realsense = realsense
        self.which_robot = which_robot

    def capture_frames(self, cameras, save_path=None):
        # Ensure save directory exists
        if save_path is not None and not os.path.exists(save_path):
            os.makedirs(save_path)

        # Let the user set the marker in the gripper
        input("Press enter when the marker is set in the gripper to grasp...")
        self.zeus.grasp(enable=True, robot=[self.which_robot])
        time.sleep(2)

        # Start camera pipeline once
        self.realsense.start_pipeline(cameras=cameras)

        # List of robot joint poses
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

        for idx, joint_state in enumerate(robot_joint_states):
            print(f"Moving to joint state {idx+1}/{len(robot_joint_states)}: {joint_state}")
            self.zeus.move_to_joint_pose(lightning_joint_pose=joint_state)
            time.sleep(2)  # Wait for robot to reach pose

            # Save RGB frame
            if save_path is not None:
                frame_filename = os.path.join(
                    save_path,
                    f"rgb_frame_{cameras[0]}_pose{idx+1}.png"
                )
            else:
                frame_filename = None

            self.realsense.save_rgb_frame(cameras=cameras, path=frame_filename)

            # Save eef poses
            eef_poses = self.zeus.get_current_eef_poses()
            eef_pose = eef_poses[self.which_robot]
            np.savetxt(
                os.path.join('eef_poses', f"eef_pose_{self.which_robot}_pose{idx+1}.txt"),
                np.array(eef_pose)
            )

        # Stop camera pipeline
        self.realsense.stop_pipeline()

    def calibrate(self, path_to_frames=None, path_to_eef_poses=None):
        # Placeholder for calibration logic
        if path_to_frames is None or path_to_eef_poses is None:
            print("Paths to frames and eef poses must be provided for calibration.")
            return
        


# ---------------- Main ----------------
if __name__ == "__main__":
    zeus = zeusRobot(robots=['lightning'])
    zeus.go_home()
    time.sleep(1)

    realsense = RealSenseCamera()
    eye_to_hand = eyeToHand(zeus, realsense, which_robot='lightning')

    cameras = ['f1380660']
    eye_to_hand.capture_frames(cameras=cameras, save_path='frames')
