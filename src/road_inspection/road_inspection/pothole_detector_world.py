import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from ultralytics import YOLO
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class YOLOv8ROS2(Node):
    def __init__(self):
        super().__init__('yolov8_inference_node')
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, '/limo/depth_camera_link/image_raw', self.image_callback, 10)
        self.model = YOLO('/home/kasm-user/inspection-task/custom (copy 1)/yolov8s_best.pt')
        self.object_count = 0

        # try:
        # image_colour = self.bridge.imgmsg_to_cv2(self.image_sub, "bgr8")
        # results = self.model(image_colour)  # results list
        # except Exception as e:
        #     print(e)
        # Run inference on an image

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            result = self.model(cv_image, save=True)
            result_image = result[0].plot()            
            display_image(None,result_image)
            self.object_count += len(result)
            self.get_logger().info(f"Detected {len(result)} objects. Total count: {self.object_count}")
        except Exception as e:
            self.get_logger().error(f"Error converting image: {e}")

    # def run_inference(self, image):
    #     # Perform inference using YOLOv8
    #     result = self.model(image, save=True)  # generator of results object
    #     return result


def display_image(path=None, image=None):
    if image is None:
        # Reading an image in default mode
        image = cv2.imread(path)

    # Check if the image is loaded successfully
    if image is None:
        print(f"Error: Unable to load image from {path}")
        return
    
     # Window name in which image is displayed 
    window_name = 'Image'
    
    # Start coordinate, here (5, 5) 
    # represents the top left corner of rectangle 
    #start_point = (5, 5) 
    
    # Ending coordinate, here (220, 220) 
    # represents the bottom right corner of rectangle 
    #end_point = (220, 220) 
    
    # Blue color in BGR 
    #color = (255, 0, 0) 
    
    # Line thickness of 2 px 
   # thickness = 2
    
    # Using cv2.rectangle() method 
    # Draw a rectangle with blue line borders of thickness of 2 px 
   # cv2.rectangle(image, start_point, end_point, color, thickness) 

  
    # Using cv2.imshow() method
    # Displaying the image
    cv2.imshow(window_name, image)

    # waits for the user to press any key
    # (this is necessary to avoid Python kernel from crashing)
    cv2.waitKey(10)

    # closing all open windows
    # cv2.destroyAllWindows()
  


def main(args=None):
    rclpy.init(args=args)
    yolo_node = YOLOv8ROS2()
    # image_path = '/home/kasm-user/ros2_ws/src/road_inspection/runs/detect/predict/image0.jpg'
    # display_image(image_path)
    rclpy.spin(yolo_node)
    yolo_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
