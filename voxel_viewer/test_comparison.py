#!/usr/bin/env python3
"""Test comparison visualization with two MarkerArrays."""

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
import time


class ComparisonTestPublisher(Node):
    """Test publisher for comparison visualization."""

    def __init__(self):
        """Initialize test publisher."""
        super().__init__('comparison_test_publisher')
        
        # Create publishers
        self.occupied_pub = self.create_publisher(
            MarkerArray, 'occupied_voxel_markers', 10
        )
        self.pattern_pub = self.create_publisher(
            MarkerArray, 'pattern_markers', 10
        )
        
        # Publish test data after 1 second
        self.timer = self.create_timer(1.0, self.publish_test_data)
        self.published = False
        
        self.get_logger().info('Comparison test publisher started')

    def publish_test_data(self):
        """Publish test marker arrays once."""
        if self.published:
            return
            
        # Create occupied voxel markers (10x10x10 grid)
        occupied_array = MarkerArray()
        occupied_marker = Marker()
        occupied_marker.type = Marker.CUBE_LIST
        occupied_marker.scale.x = 0.1
        occupied_marker.scale.y = 0.1
        occupied_marker.scale.z = 0.1
        
        # Create a 10x10x10 grid
        for x in range(10):
            for y in range(10):
                for z in range(10):
                    point = Point()
                    point.x = x * 0.1
                    point.y = y * 0.1
                    point.z = z * 0.1
                    occupied_marker.points.append(point)
                    
                    color = ColorRGBA()
                    color.r = 0.5
                    color.g = 0.5
                    color.b = 0.5
                    color.a = 1.0
                    occupied_marker.colors.append(color)
        
        occupied_array.markers.append(occupied_marker)
        
        # Create pattern markers (8x8x8 grid with offset and partial overlap)
        pattern_array = MarkerArray()
        pattern_marker = Marker()
        pattern_marker.type = Marker.CUBE_LIST
        pattern_marker.scale.x = 0.1
        pattern_marker.scale.y = 0.1
        pattern_marker.scale.z = 0.1
        
        # Create an 8x8x8 grid with partial overlap
        for x in range(2, 10):  # Offset by 2, partial overlap
            for y in range(2, 10):
                for z in range(2, 10):
                    point = Point()
                    point.x = x * 0.1
                    point.y = y * 0.1
                    point.z = z * 0.1
                    pattern_marker.points.append(point)
                    
                    color = ColorRGBA()
                    color.r = 0.5
                    color.g = 0.5
                    color.b = 0.5
                    color.a = 1.0
                    pattern_marker.colors.append(color)
        
        pattern_array.markers.append(pattern_marker)
        
        # Publish both arrays
        self.occupied_pub.publish(occupied_array)
        self.get_logger().info(f'Published occupied voxels: {len(occupied_marker.points)} points')
        
        time.sleep(0.1)  # Small delay between publishes
        
        self.pattern_pub.publish(pattern_array)
        self.get_logger().info(f'Published pattern markers: {len(pattern_marker.points)} points')
        
        self.published = True
        self.get_logger().info('Test data published. Comparison should show:')
        self.get_logger().info('- Green: overlapping region (8x8x8 = 512 points)')
        self.get_logger().info('- Red: non-overlapping regions')


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)
    
    try:
        node = ComparisonTestPublisher()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()