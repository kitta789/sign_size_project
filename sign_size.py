import cv2
import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, NamedTuple
from enum import Enum

class IphoneLens(Enum):
    WIDE = "Wide"
    ULTRA_WIDE = "Ultra Wide"

class CameraSpecs(NamedTuple):
    focal_length_mm: float
    sensor_height_mm: float
    sensor_width_mm: float
    f_number: float

# iPhone 11 camera specifications
IPHONE_11_SPECS: Dict[IphoneLens, CameraSpecs] = {
    IphoneLens.WIDE: CameraSpecs(
        focal_length_mm=26,    # 26mm equivalent focal length
        sensor_height_mm=4.8,  # Approximate sensor height
        sensor_width_mm=6.4,   # Approximate sensor width
        f_number=1.8,          # f/1.8 aperture
    ),
    IphoneLens.ULTRA_WIDE: CameraSpecs(
        focal_length_mm=13,    # 13mm equivalent focal length
        sensor_height_mm=4.8,  # Approximate sensor height
        sensor_width_mm=6.4,   # Approximate sensor width
        f_number=2.4,          # f/2.4 aperture
    ),
}

@dataclass
class Measurement:
    points: List[Tuple[int, int]]
    width: float
    height: float
    depth: float

class ObjectMeasurement:
    def __init__(self, lens_type: IphoneLens = IphoneLens.WIDE):
        self.set_camera_specs(lens_type)
        self.points = []
        self.measurements: List[Measurement] = []
        self.image = None
        self.depth_map = None
        self.window_name = f"iPhone 11 {lens_type.value} Lens Measurement"
    
    def set_camera_specs(self, lens_type: IphoneLens):
        specs = IPHONE_11_SPECS[lens_type]
        self.focal_length_mm = specs.focal_length_mm
        self.sensor_height_mm = specs.sensor_height_mm
        self.sensor_width_mm = specs.sensor_width_mm
        self.f_number = specs.f_number
        self.lens_type = lens_type

    def load_depth_model(self):
        self.midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
        self.transform = midas_transforms.small_transform
        
        self.midas.to(self.device)
        self.midas.eval()

    def estimate_depth(self, img):
        input_batch = self.transform(img).to(self.device)
        
        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        depth_map = prediction.cpu().numpy()
        
        # Normalize depth map to more realistic values for iPhone (in meters)
        # These values are approximated and may need adjustment
        min_depth = 0.3  # 0.3 meters (close focus distance)
        max_depth = 10.0 # 10 meters (approximate far distance)
        
        depth_map = np.clip(depth_map, depth_map.min(), depth_map.max())
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        depth_map = depth_map * (max_depth - min_depth) + min_depth
        
        return depth_map

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            
            if len(self.points) == 4:
                self.calculate_measurement()
                self.points = []
        
        self.draw_interface()

    def calculate_measurement(self):
        if len(self.points) != 4:
            return

        # Calculate width and height in pixels
        top_left, top_right, bottom_right, bottom_left = self.points
        
        # Calculate width (average of top and bottom)
        width_top = np.sqrt((top_right[0] - top_left[0])**2 + (top_right[1] - top_left[1])**2)
        width_bottom = np.sqrt((bottom_right[0] - bottom_left[0])**2 + (bottom_right[1] - bottom_left[1])**2)
        pixel_width = (width_top + width_bottom) / 2

        # Calculate height (average of left and right)
        height_left = np.sqrt((bottom_left[0] - top_left[0])**2 + (bottom_left[1] - top_left[1])**2)
        height_right = np.sqrt((bottom_right[0] - top_right[0])**2 + (bottom_right[1] - top_right[1])**2)
        pixel_height = (height_left + height_right) / 2

        # Calculate center point of the object
        center_x = int(np.mean([p[0] for p in self.points]))
        center_y = int(np.mean([p[1] for p in self.points]))

        # Get depth at center point
        depth = self.depth_map[center_y, center_x]

        # Calculate real-world dimensions
        img_height, img_width = self.image.shape[:2]
        
        # Calculate actual focal length in pixels
        focal_length_pixels_height = (self.focal_length_mm * img_height) / self.sensor_height_mm
        focal_length_pixels_width = (self.focal_length_mm * img_width) / self.sensor_width_mm
        
        # Use average focal length for calculation
        focal_length_pixels = (focal_length_pixels_height + focal_length_pixels_width) / 2
        
        real_width = (pixel_width * depth) / focal_length_pixels
        real_height = (pixel_height * depth) / focal_length_pixels

        measurement = Measurement(
            points=self.points.copy(),
            width=real_width,
            height=real_height,
            depth=depth
        )
        self.measurements.append(measurement)

    def draw_interface(self):
        display_img = self.image.copy()

        # Draw all completed measurements
        for idx, measurement in enumerate(self.measurements):
            self.draw_measurement(display_img, measurement, idx)

        # Draw current points
        for i, point in enumerate(self.points):
            cv2.circle(display_img, point, 3, (0, 255, 0), -1)
            if i > 0:
                cv2.line(display_img, self.points[i-1], point, (0, 255, 0), 2)
        
        if len(self.points) == 4:
            cv2.line(display_img, self.points[-1], self.points[0], (0, 255, 0), 2)

        # Draw camera info
        camera_info = f"iPhone 11 {self.lens_type.value} Lens (f/{self.f_number}, {self.focal_length_mm}mm)"
        cv2.putText(display_img, camera_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow(self.window_name, display_img)

    def draw_measurement(self, img, measurement: Measurement, index: int):
        # Draw the quadrilateral
        for i in range(4):
            cv2.line(img, measurement.points[i], measurement.points[(i+1)%4], (0, 255, 0), 2)

        # Calculate center point for text
        center_x = int(np.mean([p[0] for p in measurement.points]))
        center_y = int(np.mean([p[1] for p in measurement.points]))

        # Draw measurement information
        text = f"#{index+1} W: {measurement.width:.2f}m H: {measurement.height:.2f}m D: {measurement.depth:.2f}m"
        cv2.putText(img, text, (center_x - 60, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    def process_image(self, image_path: str):
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not load image from {image_path}")
            
        img_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        
        # Estimate depth
        self.depth_map = self.estimate_depth(img_rgb)
        
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        self.draw_interface()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.points = []
                self.measurements = []
                self.draw_interface()
            elif key == ord('w'):
                self.set_camera_specs(IphoneLens.WIDE)
                self.draw_interface()
            elif key == ord('u'):
                self.set_camera_specs(IphoneLens.ULTRA_WIDE)
                self.draw_interface()

        cv2.destroyAllWindows()
        return self.measurements

def main():
    image_path = "./data/test2.jpg"
    
    print("iPhone 11 Object Measurement Tool")
    print("\nInstructions:")
    print("1. Click on the 4 corners of the object in this order:")
    print("   Top-Left -> Top-Right -> Bottom-Right -> Bottom-Left")
    print("2. After 4 clicks, the measurement will be calculated")
    print("3. You can measure multiple objects")
    print("\nKeyboard controls:")
    print("'w' - Switch to Wide lens mode")
    print("'u' - Switch to Ultra Wide lens mode")
    print("'r' - Reset all measurements")
    print("'q' - Quit")
    
    try:
        measurer = ObjectMeasurement()
        measurer.load_depth_model()
        measurements = measurer.process_image(image_path)
        
        print("\nMeasurements:")
        for idx, m in enumerate(measurements):
            print(f"Object #{idx+1}:")
            print(f"  Width: {m.width:.2f} meters")
            print(f"  Height: {m.height:.2f} meters")
            print(f"  Depth: {m.depth:.2f} meters")
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()