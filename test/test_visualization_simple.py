#!/usr/bin/env python3
"""Simple test to verify Open3D visualization works correctly."""

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
import time


class TestPublisher(Node):
    """Test publisher for marker arrays."""

    def __init__(self):
        """Initialize test publisher."""
        super().__init__('test_publisher')
        
        # Create publishers
        self.pattern_pub = self.create_publisher(
            MarkerArray, 'pattern_markers', 10
        )
        
        # Timer for periodic publishing
        self.timer = self.create_timer(2.0, self.publish_test_data)
        
        self.get_logger().info('Test publisher started')

    def publish_test_data(self):
        """Publish test marker array."""
        marker_array = MarkerArray()
        
        # Create a CUBE_LIST marker with test points
        marker = Marker()
        marker.type = Marker.CUBE_LIST
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        
        # Create a 10x10x10 grid of points
        for x in range(10):
            for y in range(10):
                for z in range(10):
                    point = Point()
                    point.x = x * 0.2
                    point.y = y * 0.2
                    point.z = z * 0.2
                    marker.points.append(point)
                    
                    color = ColorRGBA()
                    color.r = x / 10.0
                    color.g = y / 10.0
                    color.b = z / 10.0
                    color.a = 1.0
                    marker.colors.append(color)
        
        marker_array.markers.append(marker)
        
        self.pattern_pub.publish(marker_array)
        self.get_logger().info(f'Published {len(marker.points)} test points')


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)
    
    try:
        node = TestPublisher()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()