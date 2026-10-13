# Data Viewer

This folder contains the data viewer application, which is a separate component from the main research website.

## Structure

```
data_viewer/
├── index.html                 # Redirects to data_viewer_page.html
├── data_viewer_page.html      # Main data viewer interface
├── README.md                  # This file
└── static/
    ├── css/
    │   └── bulma.min.css      # Bulma CSS framework
    ├── images/
    │   └── favicon.svg        # Website favicon
    └── data/                  # All task data and media
        ├── task-index.json    # Master index of all tasks
        ├── tasks/             # Individual task JSON files
        ├── videos/            # Task video files
        └── point-clouds/      # Point cloud data files
```

## Usage

1. **Standalone**: Navigate to this folder and serve it with a local HTTP server
   ```bash
   cd data_viewer
   python3 -m http.server 8080
   ```
   Then visit: http://localhost:8080

2. **From main website**: The main website can link to this data viewer

## Features

- **Category-based navigation**: Browse tasks by category (Pick and Place, Tool Use, etc.)
- **5-section display**: View Text Description, Egocentric View, Third View, Point Cloud, and Task ID
- **Dynamic loading**: Tasks are loaded dynamically from JSON files
- **Responsive design**: Works on desktop and mobile devices

## Data Structure

Each task is defined by a JSON file containing:
- `task_id`: Unique identifier
- `task_name`: Human-readable name
- `category`: Task category
- `text_description`: Detailed description
- `egocentric_video`: Path to first-person view video
- `third_view_video`: Path to third-person view video
- `point_cloud`: Path to point cloud data
- `difficulty`: Task difficulty level
- `duration`: Estimated duration
- `tags`: Array of relevant tags

## Adding New Tasks

1. Create a new JSON file in the appropriate category folder under `static/data/tasks/`
2. Add the task to `static/data/task-index.json`
3. Place video files in the corresponding folder under `static/data/videos/`
4. Place point cloud files in the corresponding folder under `static/data/point-clouds/`
