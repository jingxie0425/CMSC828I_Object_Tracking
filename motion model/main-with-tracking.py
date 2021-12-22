import pathlib
import argparse
import numpy as np
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
pwd = pathlib.Path.cwd()
left_images_path = pwd.joinpath('left_images') # pwd/'left_images'
right_images_path = pwd.joinpath('right_images') # pwd/'left_images'


#print(type(pwd))
#print(right_images_path)

######################################## GOOD REFERENCES ########################################
# 1. How to set diparity values: https://learnopencv.com/depth-perception-using-stereo-camera-python-c/
# 2. What does Q matrix mean: https://answers.opencv.org/question/187734/derivation-for-perspective-transformation-matrix-q/?answer=187997#post-id-187997
# 3. stereo rectification: https://www.andreasjakl.com/understand-and-apply-stereo-rectification-for-depth-maps-part-2/
# 4. projection matrix description: https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#details
# 5. Types of tracker: https://learnopencv.com/object-tracking-using-opencv-cpp-python/

"""
Tracking approach:
map {}
tracked_ids = set()
If first frame->just save the vehicles
    map{0: vehicle, 1:vehicle, .......}

next frame:
    get vehicles
    for each vehicle:
        compare which vehicle is in map (maybe try euclidean distance and other metrics)
        if match found:
            map[id] = update_measured_state
            add to tracked ids (local ones)
        else
            map register new vehicle
    for each element in map:
        if id is not in tracked ids:
            mark as tracking lost

            
"""
class Vehicle:
    def __init__(self, bbox, gray_image, point_cloud):
        self.bbox = bbox
        self.predicted_bbbox = None
        self.gray_image = gray_image
        self.point_cloud = point_cloud
        self.id = None
    
    def set_id(self, id):
        self.id = id
    
    def set_bbox(self, bbox):
        self.bbox

    def set_gray_image(self, gray_image):
        self.gray_image = gray_image

    def set_depth_map(self, depth_map):
        self.depth_map = depth_map

    def set_point_cloud(self, point_cloud):
        self.point_cloud = point_cloud

class StereoProcessing:
    def __init__(self, intrinsics, weight, classes, config):
        print("Stereo Processing class initiated.")
        self.scale = 0.00392
        self.width = None
        self.height = None
        self.channels = None
        self.classes = None
        self.class_file = classes
        self.weight_file = weight
        self.intrinsic_file = intrinsics
        self.config_file = config
        self.net = None
        self.blob = None
        self.output_layers = None
        self.layer_names = None
        self.left_images = None
        self.right_images = None
        self.valid_classes = [1,2,3,5,7]
        self.P0 = np.array([[2.007113e+03, 0.0, 9.113779e+02, 0.0],
                           [0.0, 2.007113e+03, 3.953267e+02, 0.0],
                           [0.0, 0.0, 1.0, 0.0]], dtype='float32')
        self.P1 = np.array([[2.007113e+03, 0.0, 9.113779e+02, -1.093101e+03],
                            [0.0, 2.007113e+03, 3.953267e+02, 0.0], 
                            [0.0, 0.0, 1.0, 0.0]], dtype='float32')

        self.K0, self.R0, self.T0 = self.decompose_projection_matrix(self.P0)
        self.K1, self.R1, self.T1 = self.decompose_projection_matrix(self.P1)

        self.Q = np.array([[1, 0, 0, -9.113779e+02],
                          [0, 1, 0, -3.953267e+02],
                          [0, 0, 0, 2.007113e+03],
                          [0, 0, -1/5.4461354e-01, 0]], dtype='float32')


        print("P0:\n",self.P0, "\nP1:\n",self.P1)
        self.classes = self.read_classes(self.class_file)
        # create a network
        self.net = cv2.dnn.readNet(str(self.weight_file), str(self.config_file))
        # get output layers
        self.layer_names = self.net.getLayerNames()
        print("Layer Names:\n", self.layer_names)
        print("Output layer indices: ", self.net.getUnconnectedOutLayers())
        # we have to substract 1 for getting correct layer.
        self.output_layers = [self.layer_names[i-1] for i in self.net.getUnconnectedOutLayers()]
        print("Output layer: ", self.output_layers)

        #### Tracking related parameters 
        self.previous_vehicles = {}
        self.tracked_ids = {} # set
        self.max_vehicles = 25
        self.frame_count = 0
        self.idcount = 0

    def track_vehicles(self, vehicles):
        if len(vehicles) == 0:
            return
        current_vehicles = {}
        print("Tracking vehicles", self.frame_count)
        # first frame
        if self.frame_count == 1:
            for vehicle in vehicles:
                vehicle.set_id(self.idcount)
                self.previous_vehicles[self.idcount] = vehicle
                self.idcount += 1
            print("Previous V: ", self.previous_vehicles)
            return

        # current at this point we have previous objects.   
        for vehicle in vehicles:
            best_id = self.get_closest_match_id(vehicle)
            if best_id is not -1:
                # cv2.imshow("prev ", self.previous_vehicles[best_id].gray_image)
                # cv2.imshow("curr ", vehicle.gray_image)
                # cv2.waitKey(0)
                current_vehicles[best_id] = vehicle
            else:
                current_vehicles[self.idcount] = vehicle
                self.idcount += 1
        
        self.previous_vehicles = current_vehicles

    def get_closest_match_id(self, current_vehicle):
        min_dist = 9999999
        best_id = -1
        for key, previous_vehicle in self.previous_vehicles.items():
            dist = self.calculate_dist_metric(current_vehicle, previous_vehicle)
            if dist < 5 and dist < min_dist:
                min_dist = dist
                best_id = key

        # print("Best id: ", best_id)
        return best_id

    def calculate_dist_metric(self, current_vehicle, previous_vehicle):
        prev_cx = (previous_vehicle.bbox[0] + previous_vehicle.bbox[2]) / 2.0
        prev_cy = (previous_vehicle.bbox[1] + previous_vehicle.bbox[3]) / 2.0

        curr_cx = (current_vehicle.bbox[0] + current_vehicle.bbox[2]) / 2.0
        curr_cy = (current_vehicle.bbox[1] + current_vehicle.bbox[3]) / 2.0

        dist = np.sqrt((curr_cx-prev_cx)**2 + (curr_cy-prev_cy)**2)
        # print("Dist: ", dist)
        # cv2.imshow("prev ", previous_vehicle.gray_image)
        # cv2.imshow("curr ", current_vehicle.gray_image)
        # cv2.waitKey(0)
        return dist

    def read_classes(self, class_file_path):
        assert class_file_path is not None
        with open(class_file_path, 'r') as f:
            self.classes = [ line.strip() for line in f.readlines()]
        print("Available classes: ", self.classes)

    def display_bounding_box(self, image, bbox, id=None, NMS=False):
        title = "NMS" if NMS else "Non NMS"
        x1 = round(bbox[0])
        y1 = round(bbox[1])
        x2 = round(bbox[0] + bbox[2])
        y2 = round(bbox[1] + bbox[3])
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (x1, y1)
        fontScale = 1
        color = (0, 0, 255)
        thickness = 2
        text = str(id)
        # image = cv2.putText(image, text, org, font, 
                        # fontScale, color, thickness, cv2.LINE_AA)
        # cv2.imshow(title, image)

    def display_left_and_right_images(self, left_img_dir, right_img_dir):
        """
        both args have type pathlib.Path
        """
        if left_img_dir is None or not left_img_dir.exists() or right_img_dir is None or not right_img_dir.exists():
            print("Invalid left or right image directory")
            return

        self.left_images = sorted(left_img_dir.iterdir())
        self.right_images = sorted(right_img_dir.iterdir())

        assert len(self.left_images) == len(self.right_images)
        for i in range(len(self.left_images)):
            img_left = cv2.imread(str(self.left_images[i]))
            img_right = cv2.imread(str(self.right_images[i]))
            img_left_gray = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
            img_right_gray = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
            assert img_left.shape == img_right.shape
            self.height, self.width, self.channels = img_left.shape
            left_bboxes, left_indices = self.detect_and_calculate_bounding_boxes(img_left)
            right_bboxes, right_indices = self.detect_and_calculate_bounding_boxes(img_right)
            # cv2.imshow("test", left_right_combined)
            # k = cv2.waitKey(0)
            disparity_map_left = self.calculate_disparity(img_left_gray, img_right_gray, img_left)
            depth_map = self.calculate_depth_map(disparity_map_left)
            assert depth_map.shape == disparity_map_left.shape
            vehicles = self.create_vehicle_objects_in_current_frame(img_left_gray, depth_map, left_bboxes)
            self.frame_count += 1
            self.track_vehicles(vehicles)
            # display each vehicle with its unique id
            for key, value in self.previous_vehicles.items():
                self.display_bounding_box(img_left, value.bbox, key)
                self.display_bounding_box(img_right, value.bbox, key)
            left_right_combined = np.concatenate((img_left, img_right), axis=1)
            cv2.imshow("Object tracking final: ", left_right_combined)
            k = cv2.waitKey(0)
            if k == 27:
                break

    def calculate_disparity(self, img_left, img_right, img_left_color):
        # try StereoBM
        # numDisparities = 16
        # stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=11)
        # disparity_map_left = stereo.compute(img_left,img_right).astype(np.float32)
        # disparity_map_left = disparity_map_left/16.0

        stereoProcessor = cv2.StereoSGBM_create(minDisparity=0, numDisparities=16*4, blockSize=11, P1=8*3*7**2, P2=32*3*7**2, uniquenessRatio=0)
        disparity_map_left = stereoProcessor.compute(img_left, img_right).astype(np.float32)/16.0
        return disparity_map_left

    def create_vehicle_objects_in_current_frame(self, img_left_gray, depth_map, left_bboxes):
        print("Creating vehicles for current frmae")
        """
        for each bbox
            get corresponding gray patch  
            get corresponding depth path
            get corresponding pointclouds
        """
        vehicles = []
        for box in left_bboxes:
            x = int(box[0])
            y = int(box[1])
            w = int(box[2])
            h = int(box[3])
            print(x,y,w,h)
            gray_patch = img_left_gray[y:y+h, x:x+w]
            depth_patch = depth_map[y:y+h, x:x+w]
            point_cloud = self.calculate_pointclouds(depth_map, x,y,w,h)
            vehicle = Vehicle(box, gray_patch, point_cloud)
            vehicles.append(vehicle)
        return vehicles


    def display_pointcloud(self, point_cloud):
        # Creating dataset
        x = []
        y = []
        z = []
        for point in point_cloud:
            if point[2] < 50:
                x.append(point[0])
                y.append(point[1])
                z.append(point[2])

        # Creating figure
        fig = plt.figure(figsize = (10, 7))
        ax = plt.axes(projection ="3d")
    
        # Creating plot
        r = np.random.randint(0,255)/255
        g = np.random.randint(0,255)/255
        b = np.random.randint(0,255)/255
        rgb = (r,g,b)
        ax.scatter3D(x, y, z, color = rgb)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        
        plt.title("simple 3D scatter plot")
        # show plot
        plt.show()
    
    def calculate_pointclouds(self, depth_map, x,y,w,h):
        # print("depth_map", depth_map)
        x_coords = [i for i in range(x,x+w)]
        y_coords = [j for j in range(y,y+h)]
        fx = self.K0[0,0]
        fy = self.K0[1,1]
        cx = self.K0[0,2]
        cy = self.K0[1,2]
        point_cloud = []
        for y in y_coords:
            for x in x_coords:
                if x < 0 or x >= depth_map.shape[1] or y < 0 or y >= depth_map.shape[0]:
                    continue 
                depth = depth_map[y,x]
                if depth == 0:
                    continue
                Xnorm = (x - cx) / fx
                X = Xnorm * depth
                Ynorm = (y - cy) / fy
                Y = Ynorm * depth
                Z = depth
                point_cloud.append([X, Y, Z])
        return point_cloud

    def calculate_depth_map(self, disparity_map):
        fx = self.K0[0,0]
        b = (self.T1[0,0] - self.T0[0,0])/10
        disparity_map[disparity_map==0] = 0.9
        disparity_map[disparity_map==-1] = 0.9
        depth_map = np.ones(disparity_map.shape, np.single)
        depth_map[:] = fx * b / disparity_map[:]
        return depth_map

    def decompose_projection_matrix(self, P):
        cameraMatrix, rotMatrix, transVect, rotMatrixX, rotMatrixY, rotMatrixZ, eulerAngles = cv2.decomposeProjectionMatrix(P)
        k = cameraMatrix
        R = rotMatrix
        T = transVect / transVect[3]
        print("K: ", k)
        print("R: ", R)
        print("T: ", T)
        return k, R, T

    def detect_and_calculate_bounding_boxes(self, image):
        """
        [blobFromImage] creates 4-dimensional blob from image. 
        Optionally resizes and crops image from center, subtract mean values, scales values by scalefactor, swap Blue and Red channels.
        """
        blob = self.get_image_blob(image)
        if blob is None:
            print("blob error")
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        ## Using dl net to calculate things:
        class_ids = []
        confidences = []
        boxes = []
        boxes_after_NMS = []
        conf_threshold = 0.5
        nms_threshold = 0.5
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.6:
                    if class_id not in self.valid_classes:
                        continue
                    center_x = int(detection[0] * self.width)
                    center_y = int(detection[1] * self.height)
                    w = int(detection[2] * self.width)
                    h = int(detection[3] * self.height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])
        
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        for i in indices:
            box = boxes[i]
            boxes_after_NMS.append(box)
            # self.display_bounding_box(image, box, True)
        # print("boxes", boxes)
        print("Class Id",class_ids)
        return boxes_after_NMS, indices
        
    def get_image_blob(self, image):
        assert self.net is not None
        # TODO: figure out what's 416 by 416 in this image
        return cv2.dnn.blobFromImage(image, self.scale, (416,416), (0,0,0), True, False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stereo pairs.')
    parser.add_argument('--left', default=None, type=pathlib.Path, help='folder containing left images.')
    parser.add_argument('--right', default=None, type=pathlib.Path, help='folder containing right images.')
    parser.add_argument('--intrinsics', default=None, help='file_containing_intrinsics')
    parser.add_argument('--weights', default=None, help='file containing weights' )
    parser.add_argument('--classes', default=None, help='File containing class names.')
    parser.add_argument('--config', default=None, help='File containing yolo weights')
    args = parser.parse_args()
    
    sp = StereoProcessing(args.intrinsics, args.weights, args.classes, args.config)
    sp.display_left_and_right_images(args.left, args.right)
    