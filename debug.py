import rtde_control
import rtde_receive

print("Connecting control...")
c = rtde_control.RTDEControlInterface("10.33.55.90")
print("Control connected")

print("Connecting receive...")
r = rtde_receive.RTDEReceiveInterface("10.33.55.90")
print("Receive connected")

print("Q:", r.getActualQ())