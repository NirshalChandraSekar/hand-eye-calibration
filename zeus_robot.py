import rtde_control
import rtde_receive
import numpy as np
import time

from gripper import RobotiqGripper


class zeusRobot:
    def __init__(self, robots=['thunder', 'lightning']):

        self.robots = robots
        
        if 'thunder' in self.robots:
            self.thunder_ip = '10.33.55.89'
            self.thunder_control = rtde_control.RTDEControlInterface(self.thunder_ip)
            self.thunder_receive = rtde_receive.RTDEReceiveInterface(self.thunder_ip)
            self.thunder_gripper = RobotiqGripper()
            self.thunder_gripper.connect(self.thunder_ip, 63352)  # Assuming port 63352 for the gripper
            self.thunder_gripper._reset()
            self.thunder_gripper.activate()
            self.thunder_gripper.set_enable(True)

        if 'lightning' in self.robots:
            self.lightning_ip = '10.33.55.90'
            self.lightning_control = rtde_control.RTDEControlInterface(self.lightning_ip)
            self.lightning_receive = rtde_receive.RTDEReceiveInterface(self.lightning_ip)
            self.lightning_gripper = RobotiqGripper()
            self.lightning_gripper.connect(self.lightning_ip, 63352)  # Assuming port 63352 for the gripper
            self.lightning_gripper._reset()
            self.lightning_gripper.activate()
            self.lightning_gripper.set_enable(True)


    def go_home(self):
        if 'thunder' in self.robots:
            self.thunder_control.moveJ([3.157637357711792, -0.5073470634273072, 0.927502457295553, -2.0310217342772425, 0.021031498908996582, 0.16951298713684082], 
                                       speed=0.5, 
                                       acceleration=0.5)
        if 'lightning' in self.robots:
            self.lightning_control.moveJ([-3.1441312471972864, -2.744110246697897, -1.9301799535751343, -0.037737922077514696, 0.0213315486907959, -0.044094387684957326], 
                                         speed=0.5, 
                                         acceleration=0.5)
            
            
    def set_free_drive(self, enable=True, robot=None):
        if robot is None:
            robot = ['thunder', 'lightning']
        if enable:
            if 'thunder' in robot and 'thunder' in self.robots:
                self.thunder_control.teachMode()
            if 'lightning' in robot and 'lightning' in self.robots:
                self.lightning_control.teachMode()
        else:
            if 'thunder' in robot and 'thunder' in self.robots:
                self.thunder_control.endTeachMode()
            if 'lightning' in robot and 'lightning' in self.robots:
                self.lightning_control.endTeachMode()

    
    def move_to_joint_pose(self, thunder_joint_pose=None, lightning_joint_pose=None, speed=0.5, acceleration=0.5):

        if thunder_joint_pose is not None and 'thunder' in self.robots:
            self.thunder_control.moveJ(thunder_joint_pose, speed=speed, acceleration=acceleration)
        if lightning_joint_pose is not None and 'lightning' in self.robots:
            self.lightning_control.moveJ(lightning_joint_pose, speed=speed, acceleration=acceleration)

        else:
            print("No joint pose provided for movement.")

        
    
    def grasp(self, enable=True, robot=None, speed=100, force=50):
        if robot is None:
            robot = ['thunder', 'lightning']
        if enable:
            if 'thunder' in robot and 'thunder' in self.robots:
                self.thunder_gripper.set(255)
            if 'lightning' in robot and 'lightning' in self.robots:
                self.lightning_gripper.set(255)
        else:
            if 'thunder' in robot and 'thunder' in self.robots:
                self.thunder_gripper.set(0)
            if 'lightning' in robot and 'lightning' in self.robots:
                self.lightning_gripper.set(0)
        

    
    def get_current_joint_poses(self):
        joint_poses = {}
        if 'thunder' in self.robots:
            joint_poses['thunder'] = self.thunder_receive.getActualQ()
        if 'lightning' in self.robots:
            joint_poses['lightning'] = self.lightning_receive.getActualQ()
        return joint_poses
    
    def get_current_eef_poses(self):
        eef_poses = {}
        if 'thunder' in self.robots:
            eef_poses['thunder'] = self.thunder_receive.getActualTCPPose()
        if 'lightning' in self.robots:
            eef_poses['lightning'] = self.lightning_receive.getActualTCPPose()
        return eef_poses
    

if __name__ == "__main__":
    robot = zeusRobot(robots=['lightning'])
    robot.go_home()



