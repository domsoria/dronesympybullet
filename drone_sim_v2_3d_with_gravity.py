from math import cos, sin, pi
import os
import pygame
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pybullet as p
import pybullet_data
import time
import itertools



action_space = ['up', 'down', 'left', 'right', 'forward', 'backward']
state_space = list(np.round(np.arange(-5, 5.1, 0.1), 1))

force_values = [0.5, 1.0, 1.5]
#action_space = list(itertools.product(force_values, repeat=4))

#state_size = 3  # x, y, z coordinate
state_size = 7  # x, y, z coordinate + 4 for orientation quaternion

action_size = len(action_space)

quadcopter_pos = [0, 0, 0]
quadcopter_path = []


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 64)
        self.fc5 = nn.Linear(64, 64)
        self.fc6 = nn.Linear(64, 64)
        self.fc7 = nn.Linear(64, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        return self.fc7(x)


class Quadricottero:
    def __init__(self, x, y, z):
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        #planeId = p.loadURDF("plane.urdf")
        
        self.droneStartPos = [x, y, z]
        self.droneStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
        self.droneId = p.loadURDF("drone_x_01.urdf", self.droneStartPos, self.droneStartOrientation)

        self.prop0_link_id = p.getLinkState(self.droneId, 0)
        self.prop1_link_id = p.getLinkState(self.droneId, 1)
        self.prop2_link_id = p.getLinkState(self.droneId, 2)
        self.prop3_link_id = p.getLinkState(self.droneId, 3)
        self.numJoints = p.getNumJoints(self.droneId)

        # Initialize synthetic camera parameters
        self.camera_target_position = p.getBasePositionAndOrientation(self.droneId)[0]
        self.camera_distance = 10
        self.camera_yaw = 0
        self.camera_pitch = -40
        self.camera_roll = 0
        self.camera_up = [0, 0, 1]
        self.fov = 60  # Field of view


    def muovi_drone(self, f1, f2, f3, f4):
        p.applyExternalForce(self.droneId, 0, [0, 0, f1], [0, 0, 0], p.LINK_FRAME)
        p.applyExternalForce(self.droneId, 1, [0, 0, f2], [0, 0, 0], p.LINK_FRAME)
        p.applyExternalForce(self.droneId, 2, [0, 0, f3], [0, 0, 0], p.LINK_FRAME)
        p.applyExternalForce(self.droneId, 3, [0, 0, f4], [0, 0, 0], p.LINK_FRAME)
        p.stepSimulation()
    
    
    def get_synthetic_camera_image(self,dronePosition):
        # Assume camera's width and height are both 320
        width = height = 320
        # Assume the camera is centered at the drone's position
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=dronePosition,
            distance=1.0,
            yaw=0,   # No rotation around the drone
            pitch=0,   # No rotation around the drone
            roll=0,   # No rotation around the drone
            upAxisIndex=2)
        # Assume field of view is 60 degrees, and near and far planes are 0.01 and 100
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=float(width)/height,
            nearVal=0.01, farVal=100)
        (_, _, rgb, _, _) = p.getCameraImage(
            width=width, height=height, viewMatrix=view_matrix,
            projectionMatrix=projection_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        return rgb

collision_count = 0

def calculate_leveling_reward(orientation_quat):
    # Convert the quaternion into a rotation matrix
    rot_mat = p.getMatrixFromQuaternion(orientation_quat)

    # Reshape the rotation matrix
    rot_mat = np.array(rot_mat).reshape(3, 3)

    # Global z-axis
    global_z = np.array([0, 0, 1])

    # Drone's z-axis
    drone_z = np.dot(rot_mat, global_z)

    # Compute the dot product between the global z-axis and the drone's z-axis
    dot_product = np.dot(global_z, drone_z)

    # Compute the angle between the global z-axis and the drone's z-axis
    angle = np.arccos(dot_product)

    # Compute the rewahrd
    if ( 0.0 <=np.rad2deg(angle) <= 10.0):
        #print("The drone is leveled")
        reward = 10000  # The drone is under controll
    else:
        scaled_angle = (np.rad2deg(angle) - 5) / (90 - 5)
        
        reward = -10000 * scaled_angle  # The reward is negative and proportional to the scaled angle
        #print("The drone is not level",np.rad2deg(angle),angle,reward)
    return reward


def calcola_reward(old_state_pos, new_state_pos, target):
    global collision_count  # Ensure we have access to the global collision_count
    old_distance = np.sqrt(np.sum((np.array(old_state_pos) - np.array(target))**2))
    new_distance = np.sqrt(np.sum((np.array(new_state_pos) - np.array(target))**2))

    print ("Old distance: ", old_distance)
    print ("New distance: ", new_distance)
    
    if is_colliding(new_state_pos, target):
        print ("COLLISION DETECTED")
        collision_count += 1  # Increment the collision count
        return 50000  # Positive reward for reaching the target
    elif new_distance < old_distance:
        #print ("Avvicinamento al target")
        return 1000 * np.exp(-new_distance)  # Positive reward proportional to the inverse of the distance from the target
    else:
        #print ("Allontanamento dal target")
        return -1000 * np.exp(-new_distance)  # Negative reward proportional to the inverse of the distance from the target



def choose_action(state, epsilon):
    state = torch.tensor(state, dtype=torch.float32)
    if np.random.uniform(0, 1) < epsilon:
        print ("Random action")
        action_index = np.random.choice(len(action_space))
    else:
        #print ("Greedy action")
        with torch.no_grad():
            action_index = torch.argmax(qnetwork(state)).item()
    return action_space[action_index], action_index

def update_target_q_network():
    target_qnetwork.load_state_dict(qnetwork.state_dict())

def update_q_network_old(old_state, action_index, reward, new_state):
    old_state = torch.tensor(old_state, dtype=torch.float32)
    new_state = torch.tensor(new_state, dtype=torch.float32)
    reward = torch.tensor(reward, dtype=torch.float32)

    current_q = qnetwork(old_state)[action_index]

    # Use the main Q network to choose the action
    with torch.no_grad():
        next_action = torch.argmax(qnetwork(new_state)).item()

    # Use the target Q network to calculate the target Q value
    with torch.no_grad():
        max_new_q = target_qnetwork(new_state)[next_action].item()
        
    target_q = reward + 0.95 * max_new_q  # 0.95 is the discount factor γ

    loss = torch.square(target_q - current_q)

    #print(loss.item() , "LOSS") # Print the loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

batch_size = 1024  # Definisci la dimensione del tuo batch

# Crea delle liste vuote per accumulare i dati del batch
states = []
actions = []
rewards = []
new_states = []

def update_q_network(states, actions, rewards, new_states):
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.int64)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    new_states = torch.tensor(new_states, dtype=torch.float32)

    current_q = qnetwork(states)[range(len(states)), actions]

    with torch.no_grad():
        max_new_q = target_qnetwork(new_states).max(dim=1)[0]
        
    target_q = rewards + 0.95 * max_new_q  # 0.95 is the discount factor γ

    loss = torch.square(target_q - current_q).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()  # Return the loss



def draw_target(position, radius, color):
    visualShapeId = p.createVisualShape(shapeType=p.GEOM_SPHERE, 
                                        radius=radius, 
                                        rgbaColor=color)
    
    multiBodyId = p.createMultiBody(baseMass=0, 
                                    basePosition=position, 
                                    baseVisualShapeIndex=visualShapeId)
    return multiBodyId

def move_target(multiBodyId,new_position,nuova_orientazione):
    p.resetBasePositionAndOrientation(bodyUniqueId=multiBodyId, posObj=new_position, ornObj=nuova_orientazione)

    #global target_id  # Ensure we have access to the global target_id
    #p.removeBody(target_id)  # Remove the old target
    #target_id = draw_target(new_position, 0.1, [1, 0, 0, 1])  # Create a new target at the new position


def is_colliding(pos1, pos2, tolerance=0.2):
    return abs(pos1[0] - pos2[0]) < tolerance and abs(pos1[1] - pos2[1]) < tolerance and abs(pos1[2] - pos2[2]) < tolerance


f1 = f2 = f3 = f4 = 0

def action_to_forces(action, base_force=1.0, delta=0.01):
    # All propellers have a base force that keeps the drone hovering
    #f1 = f2 = f3 = f4 = base_force
    global f1, f2, f3, f4
    # Modify forces based on the action
    f1 += delta 
    f2 += 0
    f3 += delta
    f4 += delta
   
    return f1, f2, f3, f4

def action_to_forces_old(action, base_force=1.0, delta=0.001):
    # All propellers have a base force that keeps the drone hovering
    #f1 = f2 = f3 = f4 = base_force
    global f1, f2, f3, f4
    # Modify forces based on the action
    if action == 'up':
        f1 += delta
        f2 = f3 = f4 = f1
        #f2 += delta
        #f3 += delta
        #f4 += delta
    elif action == 'down':
        f1 -= -1*abs(delta)
        f2 = f3 = f4 = f1
        
        #f2 -= delta
        #f3 -= delta
        #f4 -= delta
    elif action == 'left':
        f1 += delta
        f4 = f1
        f2 -= delta
        f3 = f2
        #f3 -= delta
        f3 = f3 - delta
        #f4 += delta
    elif action == 'right':
        f1 -= delta
        f4 = f1
        f2 += delta
        f3 = f2
        #f3 += delta
        #f4 -= delta
    elif action == 'backward':
        f1 += delta
        f2 = f1
        #f2 += delta
        f3 -= delta
        f4 = f3 
        #f4 -= delta
    elif action == 'forward':
        f1 -= delta
        f2 = f1
        #f2 -= delta
        f3 += delta
        f4 = f3
        #f4 += delta
    #print (action, f1, f2, f3, f4)
    return f1, f2, f3, f4




def move_quadcopter(quadricottero, action_index,action):
    global quadcopter_pos
    action = action_space[action_index]

    f1, f2, f3, f4 = action_to_forces(action)
    quadricottero.muovi_drone(f1, f2, f3, f4)
    quadcopter_pos, quadcopter_orient = p.getBasePositionAndOrientation(quadricottero.droneId)
    quadcopter_pos = list(quadcopter_pos)
  
    p.resetDebugVisualizerCamera(cameraDistance=0.8, cameraYaw=0, cameraPitch=-45, cameraTargetPosition=[quadcopter_pos[0]+0.2, quadcopter_pos[1], quadcopter_pos[2]+0.2])

# Function to reset the drone
# Function to reset the drone
def reset_drone(quadricottero):
    global quadcopter_pos
    quadricottero.droneStartOrientation = p.getQuaternionFromEuler([0, 0, 0]) # get quaternion for no rotation
    p.resetBasePositionAndOrientation(quadricottero.droneId, quadricottero.droneStartPos, quadricottero.droneStartOrientation)
    p.stepSimulation()


# Function to check crash and assign crash reward
def check_crash(position):
    crash_reward = -10000  # Large negative reward
    #print (position)
    if position[2] <= 0:  # Assuming the z coordinate is height
        return True, crash_reward
    else:
        return False, 0
    

initial_lr = 0.01
final_lr = 0.0001
epochs = 1000000

def lr_scheduler(epoch):
    #print (initial_lr * (final_lr / initial_lr) ** (epoch / epochs))
    return initial_lr * (final_lr / initial_lr) ** (epoch / epochs)

def main():
    # Initialize physics server
    physicsClient = p.connect(p.GUI)
    global collision_count
    global quadcopter_pos
    global quadcopter_path
    global target_id

    collision_count = 0

    # Create target
    target_pos = [1, 1, 0.5]
    target_id = draw_target(target_pos, 0.1, [1, 0, 0, 1])

    quadricottero = Quadricottero(quadcopter_pos[0], quadcopter_pos[1], quadcopter_pos[2])
    i = 1

    # Initialize batch data
    states = []
    actions = []
    rewards = []
    new_states = []

    # Initialize total_loss and total_updates
    total_loss = 0
    total_updates = 0
    while (collision_count < 10):
        epsilon = 1. / i
        old_state_pos, old_state_orient = p.getBasePositionAndOrientation(quadricottero.droneId)

        old_state = list(old_state_pos) + list(old_state_orient)

        # Choose an do action
        action, action_index = choose_action(old_state, epsilon)
        move_quadcopter(quadricottero, action_index,action)


        new_state_pos, new_state_orient = p.getBasePositionAndOrientation(quadricottero.droneId)
        new_state = list(new_state_pos) + list(new_state_orient)

        synthetic_image = quadricottero.get_synthetic_camera_image(new_state_pos)

        # Check for crash
        crashed, crash_reward = check_crash(new_state_pos)
        if crashed:
            print("Crash detected!")
            r = crash_reward
            # Add to batch data
            states.append(old_state)
            actions.append(action_index)
            rewards.append(r)
            new_states.append(new_state)
            reset_drone(quadricottero)
        else:
            r = calcola_reward(old_state_pos, new_state_pos, target_pos) + calculate_leveling_reward(new_state_orient)
            # Add to batch data
            states.append(old_state)
            actions.append(action_index)
            rewards.append(r)
            new_states.append(new_state)

        # Update Q-network if batch size is met
        if len(states) >= batch_size:
            loss = update_q_network(states, actions, rewards, new_states)
            total_loss += loss
            total_updates += 1
            # Clear batch data
            states = []
            actions = []
            rewards = []
            new_states = []

        # Save model and print average loss at end of epoch
        if i % 1024 == 0:  # adjust as needed
            print(f"Saving model at epoch {i}...")
            torch.save(qnetwork.state_dict(), f"qnetwork_model.pth")
            update_target_q_network()
            scheduler.step()
            avg_loss = total_loss / total_updates
            print(f"Average loss: {avg_loss}")
            total_loss = 0
            total_updates = 0
        

        i += 1
    # Final update to include data not fitting into a full batch size
    if len(states) > 0:
        update_q_network(states, actions, rewards, new_states)



if __name__ == "__main__":
    qnetwork = QNetwork(state_size, action_size)
    target_qnetwork = QNetwork(state_size, action_size)

    optimizer = optim.Adam(qnetwork.parameters(), lr=initial_lr)
    # Crea lo scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_scheduler)





    model_weights_path = "model_weights.pth"  # Modificare con il percorso effettivo dei pesi del modello

    if os.path.exists(model_weights_path):
        qnetwork.load_state_dict(torch.load(model_weights_path))
        print("Loaded weights from file")
    else:
        print("No weights file found, starting from scratch")

    main()
