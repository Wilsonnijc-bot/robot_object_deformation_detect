# Task Data Structure

This folder contains all the task information for the Humanoid Everyday Data Explorer.

## Folder Structure

```
static/data/
├── task-index.json              # Master index mapping task IDs to data files
├── tasks/                       # Individual task data files organized by category
│   ├── pick-place/             # Pick and Place tasks
│   │   ├── task-1.json
│   │   ├── task-2.json
│   │   └── task-3.json
│   ├── deformable/             # Deformable Objects tasks
│   │   ├── task-1.json
│   │   ├── task-2.json
│   │   └── task-3.json
│   ├── loco-manip/             # Loco-Manipulations tasks
│   │   ├── task-1.json
│   │   ├── task-2.json
│   │   └── task-3.json
│   ├── tool-use/               # Tool Use tasks
│   │   ├── task-1.json
│   │   ├── task-2.json
│   │   └── task-3.json
│   ├── articulated/            # Articulated Object tasks
│   │   ├── task-1.json
│   │   ├── task-2.json
│   │   └── task-3.json
│   ├── high-precision/         # High Precision tasks
│   │   ├── task-1.json
│   │   ├── task-2.json
│   │   └── task-3.json
│   └── human-robot/            # Human-Robot Interaction tasks
│       ├── task-1.json
│       ├── task-2.json
│       └── task-3.json
├── videos/                     # Video files organized by category and task
│   ├── pick-place/
│   ├── deformable/
│   ├── loco-manip/
│   ├── tool-use/
│   ├── articulated/
│   ├── high-precision/
│   └── human-robot/
├── point-clouds/               # Point cloud data files
│   ├── pick-place/
│   ├── deformable/
│   ├── loco-manip/
│   ├── tool-use/
│   ├── articulated/
│   ├── high-precision/
│   └── human-robot/
└── README.md                   # This file
```

## Task Data Format

Each task JSON file contains the following fields:

```json
{
  "task_id": "category-task-number",
  "task_name": "Human-readable task name",
  "category": "Category name",
  "text_description": "Detailed description of the task",
  "egocentric_video": "Path to egocentric view video",
  "third_view_video": "Path to third-person view video",
  "point_cloud": "Path to point cloud data",
  "difficulty": "Beginner/Intermediate/Advanced",
  "duration": "Task duration in seconds",
  "tags": ["array", "of", "relevant", "tags"]
}
```

## Adding New Tasks

### 1. Create Task Data File
Create a new JSON file in the appropriate category folder:
```bash
# Example: Adding a new pick and place task
touch static/data/tasks/pick-place/task-4.json
```

### 2. Add Task Information
Fill in the task data in the JSON file:
```json
{
  "task_id": "pick-place-4",
  "task_name": "Your New Task Name",
  "category": "Pick and Place",
  "text_description": "Detailed description of your task...",
  "egocentric_video": "./static/data/videos/pick-place/task-4/egocentric.mp4",
  "third_view_video": "./static/data/videos/pick-place/task-4/third-view.mp4",
  "point_cloud": "./static/data/point-clouds/pick-place/task-4/pointcloud.ply",
  "difficulty": "Intermediate",
  "duration": "25 seconds",
  "tags": ["your", "relevant", "tags"]
}
```

### 3. Update Task Index
Add the new task to `task-index.json`:
```json
{
  "pick-place-4": "./static/data/tasks/pick-place/task-4.json"
}
```

### 4. Add Task to HTML
Add the task to the appropriate category in `data_viewer_page.html`:
```html
<div class="task-list" data-task="pick-place-4">Task 4: Your New Task Name</div>
```

## Video and Point Cloud Paths

- **Videos**: Store in `static/data/videos/category/task-number/`
- **Point Clouds**: Store in `static/data/point-clouds/category/task-number/`

## Supported File Formats

- **Videos**: MP4, WebM, OGV
- **Point Clouds**: PLY, PCD, OBJ
- **Data**: JSON

## Notes

- All paths in JSON files should be relative to the website root
- Task IDs must be unique across all categories
- The website will automatically load task data when users click on tasks
- Missing files will show placeholder text
