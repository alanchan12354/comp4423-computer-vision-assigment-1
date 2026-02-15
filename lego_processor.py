import cv2
import numpy as np

class LegoProcessor:
    def __init__(self):
        # Define palette for Task 2 (White, Gray, Black)
        # BGR format
        self.task2_palette = np.array([
            [255, 255, 255], # White
            [128, 128, 128], # Gray
            [0, 0, 0]        # Black
        ], dtype=np.uint8)

        # Define brick sizes for Task 3 (width x height)
        # Order matters: try largest first
        self.brick_sizes = [
            (4, 2), # 2x4 (w=4, h=2 usually or depends on orientation)
            (2, 4),
            (2, 2),
            (2, 1),
            (1, 2),
            (1, 1)
        ]

    def _quantize_image(self, image, palette):
        """
        Quantize the image to the nearest colors in the palette.
        """
         # Reshape image to list of pixels
        h, w, c = image.shape
        pixels = image.reshape((-1, 3))
        
        # Calculate distances to each palette color
        # This can be optimized, but for 100x100 it's fast enough
        # Broadcasting: pixels (N, 1, 3) - palette (1, M, 3)
        distances = np.linalg.norm(pixels[:, np.newaxis, :] - palette[np.newaxis, :, :], axis=2)
        
        # Get index of nearest color
        nearest_color_indices = np.argmin(distances, axis=1)
        
        # Map back to colors
        quantized_pixels = palette[nearest_color_indices]
        quantized_image = quantized_pixels.reshape((h, w, c))
        
        return quantized_image.astype(np.uint8)

    def process_task2(self, image, size=20):
        """
        Task 2: 1x1 bricks, 3 colors, max 100x100 resolution.
        """
        # 1. Resize to be small (pixel art style)
        # Max dimension 100
        h, w = image.shape[:2]
        scale = min(100.0/w, 100.0/h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        small_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # 2. Quantize colors
        quantized = self._quantize_image(small_image, self.task2_palette)

        # 3. Render 1x1 bricks
        # To make it look like LEGO, we can scale up and add a grid or "stud"
        return self._render_bricks(quantized, size=size)

    def _render_bricks(self, brick_map, size=20):
        """
        Render a pixel map as LEGO bricks.
        brick_map: (H, W, 3) array where each pixel is a brick color
        size: pixel size of one 1x1 brick in the output image
        """
        h, w = brick_map.shape[:2]
        output_h = h * size
        output_w = w * size
        canvas = np.zeros((output_h, output_w, 3), dtype=np.uint8)

        for y in range(h):
            for x in range(w):
                color = brick_map[y, x].tolist()
                
                # Draw the main brick body
                pt1 = (x * size, y * size)
                pt2 = ((x + 1) * size, (y + 1) * size)
                cv2.rectangle(canvas, pt1, pt2, color, -1)
                
                # Draw a border
                cv2.rectangle(canvas, pt1, pt2, (int(c*0.8) for c in color), 1)

                # Draw a stud on top (circle)
                center = (x * size + size // 2, y * size + size // 2)
                radius = int(size * 0.35)
                # Stud color slightly lighter/shadowed to give 3D effect
                stud_color = color # Simplify for now
                cv2.circle(canvas, center, radius, stud_color, -1)
                # Add shadow to stud
                cv2.circle(canvas, center, radius, (int(c*0.8) for c in color), 1)

        return canvas

    def process_task3(self, image, size=20):
        """
        Task 3: Multiple brick sizes, more colors.
        Returns: rendered_image, stats
        """
        # 1. Resize
        h, w = image.shape[:2]
        scale = min(60.0/w, 60.0/h) # Use slightly smaller grid for simpler rendering
        new_w = int(w * scale)
        new_h = int(h * scale)
        small_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # 2. Quantize with more colors (simple 8-level quantization per channel)
        # Or use K-means for better restricted palette.
        # Let's use a fixed extended palette for classic LEGO colors
        extended_palette = np.array([
            [255, 255, 255], # White
            [128, 128, 128], # Grey
            [0, 0, 0],       # Black
            [0, 0, 255],     # Red
            [0, 255, 0],     # Green
            [255, 0, 0],     # Blue
            [0, 255, 255],   # Yellow
            [0, 75, 150],    # Brown-ish
        ], dtype=np.uint8)
        
        quantized_grid = self._quantize_image(small_image, extended_palette) # This is HxW grid of colors
        
        # 3. Greedy placement of bricks
        # We need to cover the grid (H, W) with bricks
        # visited mask
        visited = np.zeros(quantized_grid.shape[:2], dtype=bool)
        brick_list = [] # List of (x, y, w, h, color)
        stats = {}

        grid_h, grid_w = quantized_grid.shape[:2]

        for y in range(grid_h):
            for x in range(grid_w):
                if visited[y, x]:
                    continue
                
                current_color = tuple(quantized_grid[y, x])
                
                # Try sizes
                found_brick = False
                for bw, bh in self.brick_sizes:
                    if x + bw <= grid_w and y + bh <= grid_h:
                        # Check region color match and not visited
                        region = quantized_grid[y:y+bh, x:x+bw]
                        # Check if all pixels in region match current_color
                        # Efficient check: calculate std dev or just strict equality
                        # Strict equality:
                        region_match = np.all(region == current_color)
                        
                        # Also check visited
                        if region_match and not np.any(visited[y:y+bh, x:x+bw]):
                            # Place brick
                            visited[y:y+bh, x:x+bw] = True
                            brick_list.append({
                                'x': x, 'y': y, 'w': bw, 'h': bh, 'color': current_color
                            })
                            
                            # Stats
                            key = f"{bw}x{bh}"
                            stats[key] = stats.get(key, 0) + 1
                            found_brick = True
                            break
                
                if not found_brick:
                    # Fallback to 1x1 (should be covered by loop but just in case)
                    visited[y, x] = True
                    brick_list.append({
                        'x': x, 'y': y, 'w': 1, 'h': 1, 'color': current_color
                    })
                    key = "1x1"
                    stats[key] = stats.get(key, 0) + 1

        # 4. Render from brick_list
        output_h = grid_h * size
        output_w = grid_w * size
        canvas = np.zeros((output_h, output_w, 3), dtype=np.uint8)

        for brick in brick_list:
            bx, by, bw, bh = brick['x'], brick['y'], brick['w'], brick['h']
            # Ensure color is native python int for OpenCV
            color = [int(c) for c in brick['color']]
            
            x_px = bx * size
            y_px = by * size
            w_px = bw * size
            h_px = bh * size
            
            # Brick body
            cv2.rectangle(canvas, (x_px, y_px), (x_px + w_px, y_px + h_px), color, -1)
            # Border
            cv2.rectangle(canvas, (x_px, y_px), (x_px + w_px, y_px + h_px), (0,0,0), 1)
            
            # Draw studs
            for i in range(bw):
                for j in range(bh):
                   cx = x_px + i * size + size // 2
                   cy = y_px + j * size + size // 2
                   radius = int(size * 0.35)
                   cv2.circle(canvas, (cx, cy), radius, [c*0.9 for c in color], -1)
                   cv2.circle(canvas, (cx, cy), radius, [c*0.7 for c in color], 1)

        return canvas, stats

