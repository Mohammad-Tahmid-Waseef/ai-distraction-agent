import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import threading
import time
import json
from datetime import datetime
from collections import deque
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class DistractionAgentApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Distraction Agent - Focus Tracker")
        self.root.geometry("1200x800")
        # Rolling-average graph settings
        self.smooth_window_seconds = 10
        self.recent_state_samples = deque()
        self.last_graph_update = 0.0


        
        # State variables
        self.is_running = False
        self.cap = None
        self.start_time = None
        self.end_time = None
        self.total_focus_time = 0
        self.total_distracted_time = 0
        self.last_state = "focused"
        self.last_state_change = None
        
        # Focus history (last 30 minutes)
        self.focus_history = deque(maxlen=1800)  # 30 min * 60 sec
        
        # AI Model
        self.face_mesh = None
        
        # Load saved data
        self.load_session_data()
        
        # Setup GUI
        self.setup_gui()
        
        # Initialize model in background
        threading.Thread(target=self.init_model, daemon=True).start()
    
    def init_model(self):
        """Initialize MediaPipe Face Mesh"""
        print("Loading AI model...")
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("Model loaded!")
        self.status_label.config(text="Status: Ready ✓")
    
    def setup_gui(self):
        """Setup the GUI layout"""
        # Top control panel
        control_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        control_frame.pack_propagate(False)
        
        # Start/Stop button
        self.start_button = tk.Button(
            control_frame, 
            text="▶ Start Focus Session",
            command=self.toggle_session,
            font=('Arial', 16, 'bold'),
            bg='#27ae60',
            fg='white',
            width=20,
            height=2
        )
        self.start_button.pack(side=tk.LEFT, padx=20, pady=10)
        
        # Stats display
        stats_frame = tk.Frame(control_frame, bg='#2c3e50')
        stats_frame.pack(side=tk.LEFT, padx=20)
        
        self.timer_label = tk.Label(
            stats_frame,
            text="Session: 00:00",
            font=('Arial', 18, 'bold'),
            bg='#2c3e50',
            fg='white'
        )
        self.timer_label.pack(anchor=tk.W)
        
        self.focus_label = tk.Label(
            stats_frame,
            text="Focused: 0% | Distracted: 0%",
            font=('Arial', 12),
            bg='#2c3e50',
            fg='#ecf0f1'
        )
        self.focus_label.pack(anchor=tk.W)
        
        # Status label
        self.status_label = tk.Label(
            control_frame,
            text="Status: Loading model...",
            font=('Arial', 12),
            bg='#2c3e50',
            fg='#f39c12'
        )
        self.status_label.pack(side=tk.RIGHT, padx=20)
        
        # Main content area
        content_frame = tk.Frame(self.root)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Left side - Video feed
        video_frame = tk.LabelFrame(content_frame, text="Camera Feed", font=('Arial', 12, 'bold'))
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.video_label = tk.Label(video_frame, bg='black')
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Right side - Focus graph and logs
        right_frame = tk.Frame(content_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=5)
        
        # Focus level graph
        graph_frame = tk.LabelFrame(right_frame, text="Focus Level (Last 30 min)", font=('Arial', 10, 'bold'))
        graph_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.figure = Figure(figsize=(5, 3), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_ylim(0, 1)
        self.ax.set_xlim(0, 30)
        self.ax.set_xlabel('Time (minutes ago)')
        self.ax.set_ylabel('Focus Level')
        self.ax.grid(True, alpha=0.3)
        
        self.canvas = FigureCanvasTkAgg(self.figure, graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Event log
        log_frame = tk.LabelFrame(right_frame, text="Event Log", font=('Arial', 10, 'bold'))
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.log_text = tk.Text(log_frame, height=10, font=('Courier', 9), bg='#ecf0f1')
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        scrollbar = tk.Scrollbar(self.log_text)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.log_text.yview)
        
        # Bottom buttons
        button_frame = tk.Frame(self.root)
        button_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Button(
            button_frame,
            text="📊 View Report",
            command=self.show_report,
            font=('Arial', 10),
            bg='#3498db',
            fg='white',
            width=15
        ).pack(side=tk.LEFT, padx=5)
        
        self.save_button = tk.Button(
            button_frame,
            text="💾 Save Session",
            command=self.save_session_data,
            font=('Arial', 10),
            bg='#9b59b6',
            fg='white',
            width=15,
            state=tk.DISABLED   # start disabled
        )
        self.save_button.pack(side=tk.LEFT, padx=5)

        
        tk.Button(
            button_frame,
            text="🔄 Reset Stats",
            command=self.reset_stats,
            font=('Arial', 10),
            bg='#e74c3c',
            fg='white',
            width=15
        ).pack(side=tk.LEFT, padx=5)
        
        self.log_event("Application started")
    
    def toggle_session(self):
        """Start or stop the focus session"""
        if not self.is_running:
            self.start_session()
        else:
            self.stop_session()
    
    def start_session(self):
        """Start tracking focus"""
        if self.face_mesh is None:
            messagebox.showwarning("Not Ready", "AI model is still loading. Please wait...")
            return
        
        self.is_running = True
        self.end_time = None
        self.total_focus_time = 0
        self.total_distracted_time = 0
        self.focus_history.clear()
        self.recent_state_samples.clear()
        self.last_graph_update = 0.0
        self.log_text.delete('1.0', tk.END)   # clear the log box for a fresh run
        self.log_event("Log cleared for new session")
        self.save_button.config(state=tk.DISABLED)
        self.start_time = time.time()
        self.last_state = "focused"
        self.last_state_change = self.start_time
        self.start_button.config(text="⏹ Stop Session", bg='#e74c3c')
        self.status_label.config(text="Status: Tracking...", fg='#27ae60')
        self.log_event("Focus session started")
        
        # Start camera
        self.cap = cv2.VideoCapture(0)
        
        # Start video processing thread
        threading.Thread(target=self.process_video, daemon=True).start()
        
        # Start UI update timer
        self.update_ui()
    
    def stop_session(self):
        """Stop tracking focus"""
        self.is_running = False
        self.end_time = time.time()
        self.save_button.config(state=tk.NORMAL)

        # finalize the last state time chunk
        if self.last_state_change is not None:
            current_time = self.end_time
            time_in_state = current_time - self.last_state_change

            if self.last_state == "focused":
                self.total_focus_time += time_in_state
            else:
                self.total_distracted_time += time_in_state

            self.last_state_change = current_time

        # Update focus percentage label one last time (so it matches report)
        tracked_total = self.total_focus_time + self.total_distracted_time
        if tracked_total > 0:
            focus_pct = (self.total_focus_time / tracked_total) * 100
            distract_pct = 100 - focus_pct
            self.focus_label.config(text=f"Focused: {focus_pct:.1f}% | Distracted: {distract_pct:.1f}%")
        else:
            self.focus_label.config(text="Focused: 0% | Distracted: 0%")


        # force timer label to show the final exact time
        if self.start_time is not None:
            elapsed = int(self.end_time - self.start_time)
            mins, secs = divmod(elapsed, 60)
            self.timer_label.config(text=f"Session: {mins:02d}:{secs:02d}")

        self.start_button.config(text="▶ Start Focus Session", bg='#27ae60')
        self.status_label.config(text="Status: Stopped", fg='#e74c3c')
        self.log_event("Focus session stopped")

        if self.cap:
            self.cap.release()
            self.cap = None

        self.video_label.config(image='')

    
    def process_video(self):
        """Main video processing loop"""
        while self.is_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            height, width, _ = frame.shape
            
            # Face detection and analysis
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            is_focused = True
            status_text = "No face detected"
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    pitch, yaw, roll = self.calculate_head_pose(face_landmarks, width, height)
                    is_focused, reason = self.analyze_focus(pitch, yaw)
                    
                    if is_focused:
                        status_text = "FOCUSED ✓"
                        status_color = (0, 255, 0)
                    else:
                        status_text = f"DISTRACTED - {reason}"
                        status_color = (0, 0, 255)
                    
                    # Draw status on frame
                    cv2.putText(frame, status_text, (10, 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 3)
                    
                    # Draw angles
                    angle_text = f"Pitch: {pitch:.1f}° | Yaw: {yaw:.1f}°"
                    cv2.putText(frame, angle_text, (10, height - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Draw detection thresholds as guide
                    guide_text = "Focus zone: Look straight ahead"
                    cv2.putText(frame, guide_text, (10, height - 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            else:
                is_focused = False
                cv2.putText(frame, status_text, (10, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            
            # Update state tracking
            current_time = time.time()
            if is_focused != (self.last_state == "focused"):
                time_in_state = current_time - self.last_state_change
                if self.last_state == "focused":
                    self.total_focus_time += time_in_state
                else:
                    self.total_distracted_time += time_in_state
                
                new_state = "focused" if is_focused else "distracted"
                self.log_event(f"State changed: {self.last_state} → {new_state}")
                self.last_state = new_state
                self.last_state_change = current_time
            
            # --- Rolling-average focus history (Option A) ---
            value = 1.0 if is_focused else 0.0
            now = time.time()

            # keep raw samples for the last N seconds
            self.recent_state_samples.append((now, value))
            cutoff = now - self.smooth_window_seconds
            while self.recent_state_samples and self.recent_state_samples[0][0] < cutoff:
                self.recent_state_samples.popleft()

            # add ONE point per second to the graph history
            if now - self.last_graph_update >= 1.0:
                avg_focus = sum(v for _, v in self.recent_state_samples) / max(1, len(self.recent_state_samples))
                self.focus_history.append((now, avg_focus))
                self.last_graph_update = now

            
            # Display frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((640, 480), Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
            
            time.sleep(0.03)  # ~30 FPS
    
    def calculate_head_pose(self, face_landmarks, image_width, image_height):
        """
        Calculate head pose (pitch, yaw, roll) from face landmarks
        Returns angles in degrees
        """
        # Key landmark indices for pose estimation
        landmark_indices = [1, 199, 33, 263, 61, 291]
        
        # 3D model points (approximate face model)
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye corner
            (225.0, 170.0, -135.0),      # Right eye corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ])
        
        # 2D image points from landmarks
        image_points = np.array([
            (face_landmarks.landmark[idx].x * image_width,
             face_landmarks.landmark[idx].y * image_height)
            for idx in landmark_indices
        ], dtype=np.float64)
        
        # Camera internals
        focal_length = image_width
        center = (image_width / 2, image_height / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # Assuming no lens distortion
        dist_coeffs = np.zeros((4, 1))
        
        # Solve PnP to get rotation and translation vectors
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        
        # Calculate Euler angles from rotation matrix
        pose_matrix = cv2.hconcat((rotation_matrix, translation_vector))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_matrix)
        
        pitch = euler_angles[0][0]
        yaw = euler_angles[1][0]
        roll = euler_angles[2][0]
        
        return pitch, yaw, roll
    
    def analyze_focus(self, pitch, yaw):
        """
        Determine if user is focused based on head pose
        Returns: (is_focused, reason)
        """
        # Thresholds (adjustable)
        YAW_THRESHOLD = 20      # Looking left/right
        PITCH_THRESHOLD = 15    # Looking down/up
        
        # Fix pitch inversion - normalize to -180 to 180 range
        if pitch > 90:
            pitch = pitch - 180
        elif pitch < -90:
            pitch = pitch + 180
        
        if abs(yaw) > YAW_THRESHOLD:
            return False, "Looking away (left/right)"
        if abs(pitch) > PITCH_THRESHOLD:  # Use abs() for both up and down
            if pitch > 0:
                return False, "Looking down"
            else:
                return False, "Looking up"
        
        return True, ""
    
    def update_ui(self):
        """Update UI elements"""
        if not self.is_running:
            return
        
        # Update timer
        if self.start_time:
            elapsed = int(time.time() - self.start_time)
            mins, secs = divmod(elapsed, 60)
            self.timer_label.config(text=f"Session: {mins:02d}:{secs:02d}")
        
        # Update focus percentage (use tracked time)
        tracked_total = self.total_focus_time + self.total_distracted_time
        if tracked_total > 0:
            focus_pct = (self.total_focus_time / tracked_total) * 100
            distract_pct = 100 - focus_pct
            self.focus_label.config(text=f"Focused: {focus_pct:.1f}% | Distracted: {distract_pct:.1f}%")
        else:
            self.focus_label.config(text="Focused: 0% | Distracted: 0%")

        
        # Update graph
        self.update_graph()
        
        # Schedule next update
        self.root.after(1000, self.update_ui)
    
    def update_graph(self):
        """Update the focus level graph"""
        if len(self.focus_history) == 0:
            return
        
        current_time = time.time()
        times = []
        levels = []

        for timestamp, level in self.focus_history:
            minutes_ago = (current_time - timestamp) / 60
            if minutes_ago <= 30:  # Only show last 30 minutes
                times.append(minutes_ago)
                levels.append(level)
        
        self.ax.clear()
        if times:
            pairs = sorted(zip(times, levels))
            times, levels = zip(*pairs)
            self.ax.fill_between(times, levels, alpha=0.3, color='green')
            self.ax.plot(times, levels, color='darkgreen', linewidth=2)
        
        self.ax.set_ylim(0, 1.1)
        self.ax.set_xlim(0, 30)
        self.ax.set_xlabel('Minutes Ago')
        self.ax.set_ylabel('Focus Level')
        self.ax.set_title('Focus Over Time')
        self.ax.grid(True, alpha=0.3)
        
        self.canvas.draw()
    
    def log_event(self, message):
        """Add event to log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        self.log_text.insert(tk.END, log_message)
        self.log_text.see(tk.END)
    
    def show_report(self):
        end = self.end_time if self.end_time is not None else time.time()
        total = int(end - self.start_time) if self.start_time else 0

        if total <= 0:
            messagebox.showinfo("Report", "No data yet. Start a session to track your focus!")
            return

        # Copy totals so we can safely add the current chunk (if running)
        focus_time = self.total_focus_time
        distracted_time = self.total_distracted_time

        # If session is running, include time since last_state_change
        if self.is_running and self.last_state_change is not None:
            now = time.time()
            dt = now - self.last_state_change
            if self.last_state == "focused":
                focus_time += dt
            else:
                distracted_time += dt

        tracked_total = focus_time + distracted_time
        focus_pct = (focus_time / tracked_total) * 100 if tracked_total > 0 else 0

        report = f"""
    Focus Report
    {'=' * 40}

    Total Session Time: {int(total // 60)} minutes {int(total % 60)} seconds
    Focused Time: {int(focus_time // 60)} min {int(focus_time % 60)} sec ({focus_pct:.1f}%)
    Distracted Time: {int(distracted_time // 60)} min {int(distracted_time % 60)} sec ({100-focus_pct:.1f}%)

    Events Logged: {int(self.log_text.index('end-1c').split('.')[0])}

    Tips:
    • Aim for >80% focus time
    • Take breaks every 25-30 minutes
    • Check your posture!
    """

        messagebox.showinfo("Focus Report", report)


    
    def reset_stats(self):
        """Reset all statistics"""
        if messagebox.askyesno("Reset Stats", "Are you sure you want to reset all statistics?"):
            self.total_focus_time = 0
            self.total_distracted_time = 0
            self.focus_history.clear()
            self.recent_state_samples.clear()
            self.last_graph_update = 0.0
            self.log_text.delete('1.0', tk.END)
            self.log_event("Statistics reset")
    
    def save_session_data(self):
        """Save session data to JSON file"""
        data = {
            'timestamp': datetime.now().isoformat(),
            'total_focus_time': self.total_focus_time,
            'total_distracted_time': self.total_distracted_time,
            'focus_history': list(self.focus_history)
        }
        
        filename = f'focus_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.log_event(f"Session data saved to {filename}")
        messagebox.showinfo("Saved", f"Session data saved to:\n{filename}")
    
    def load_session_data(self):
        """Load previous session data if exists"""
        try:
            # Try to load the most recent focus_log file
            import glob
            files = glob.glob('focus_log_*.json')
            if files:
                latest_file = max(files)
                with open(latest_file, 'r') as f:
                    data = json.load(f)
                    print(f"Previous session data loaded from {latest_file}")
        except Exception as e:
            print(f"No previous session data found: {e}")

def main():
    root = tk.Tk()
    app = DistractionAgentApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.stop_session(), root.destroy()))
    root.mainloop()

if __name__ == "__main__":
    main()
