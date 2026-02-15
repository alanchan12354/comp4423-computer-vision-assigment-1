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

    def _draw_stud(self, canvas, cx, cy, size, color):
        """
        Draws a single LEGO stud with 3D shading effects.
        """
        r = int(size * 0.35)
        if r < 1: r = 1
        
        # Colors
        # Ensure color is a list of ints
        base_c = [int(c) for c in color]
        
        # Calculate shadow/light colors (clamped)
        shadow_c = [max(0, int(c * 0.6)) for c in base_c]
        dark_c   = [max(0, int(c * 0.8)) for c in base_c]
        light_c  = [min(255, int(c * 1.4)) for c in base_c]
        
        # 1. Drop Shadow (on the plate/brick below)
        # Shifted bottom-right
        offset = max(1, int(size * 0.08))
        cv2.circle(canvas, (cx + offset, cy + offset), r, tuple(shadow_c), -1)
        
        # 2. Main Stud Body
        cv2.circle(canvas, (cx, cy), r, tuple(base_c), -1)
        
        # 3. 3D Bevel/Highlight
        # Using arcs to simulate light coming from Top-Left
        
        # Highlight (Top-Left 135 to 315 degrees? No, OpenCV angles: 0=Right, 90=Down)
        # Top-Left is -135 or 225.
        # We want an arc from approx 180 (Left) to 270 (Top) centered at Top-Left.
        # Let's draw a lighter crescent on top-left
        cv2.ellipse(canvas, (cx, cy), (r, r), 0, 180, 270, tuple(light_c), max(1, int(size*0.1)))
        
        # Shadow (Bottom-Right)
        # Arc from 0 (Right) to 90 (Down)
        cv2.ellipse(canvas, (cx, cy), (r, r), 0, 0, 90, tuple(dark_c), max(1, int(size*0.1)))
        
        # Extra Specular Highlight dot
        spec_r = max(1, int(r * 0.3))
        spec_x = cx - int(r * 0.4)
        spec_y = cy - int(r * 0.4)
        cv2.circle(canvas, (spec_x, spec_y), spec_r, (255, 255, 255), -1, cv2.LINE_AA)

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
                
                # Coordinates
                x0 = x * size
                y0 = y * size
                x1 = (x + 1) * size
                y1 = (y + 1) * size
                
                # 1. Draw Brick Plate
                cv2.rectangle(canvas, (x0, y0), (x1, y1), color, -1)
                
                # 2. Draw Brick Border (Grid effect)
                # Darken the border slightly or use black for contrast
                # Reference image has defined edges.
                border_color = tuple([max(0, int(c * 0.7)) for c in color])
                cv2.rectangle(canvas, (x0, y0), (x1, y1), border_color, 1)

                # 3. Draw Stud
                cx = x0 + size // 2
                cy = y0 + size // 2
                self._draw_stud(canvas, cx, cy, size, color)

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
            
            # 1. Brick body
            cv2.rectangle(canvas, (x_px, y_px), (x_px + w_px, y_px + h_px), color, -1)
            
            # 2. Border
            border_color = tuple([max(0, int(c * 0.7)) for c in color])
            cv2.rectangle(canvas, (x_px, y_px), (x_px + w_px, y_px + h_px), border_color, 1)
            
            # 3. Draw studs
            for i in range(bw):
                for j in range(bh):
                   cx = x_px + i * size + size // 2
                   cy = y_px + j * size + size // 2
                   self._draw_stud(canvas, cx, cy, size, color)

        return canvas, stats

