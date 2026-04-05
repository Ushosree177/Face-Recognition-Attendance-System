import cv2

DATASET_DIR    = "data/dataset"    
TRAINER_DIR    = "data/trainer"    
UNKNOWN_DIR    = "data/unknown"    
ATTENDANCE_DIR = "data/attendance" 

CASCADE_PATH         = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
DETECT_SCALE_FACTOR  = 1.2
DETECT_MIN_NEIGHBORS = 5
DETECT_MIN_SIZE      = (60, 60)

IMG_SIZE               = (200, 200)          
DEFAULT_CONF_THRESHOLD = 70                  
MODEL_FILE             = "model.yml"
LABEL_FILE             = "labels.pkl"

DEFAULT_CAPTURE_COUNT  = 100                 
CAPTURE_MIN            = 30
CAPTURE_MAX            = 250

UNKNOWN_SAVE_COOLDOWN  = 3.0                 

CAMERA_INDEX           = 1                  
CAMERA_TICK_MS         = 33                  

C_BG       = "#0d0d14"
C_PANEL    = "#13131e"
C_SURFACE  = "#1c1c2e"
C_BORDER   = "#2a2a40"
C_ACCENT   = "#7c6af7"   
C_ACCENT2  = "#00e5c3"   
C_RED      = "#f75a7c"
C_WARN     = "#f7b25a"
C_SUCCESS  = "#5af7a0"
C_TEXT     = "#e8e8f0"
C_TEXT_DIM = "#6e6e8a"