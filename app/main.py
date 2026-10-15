from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import point_ops, color, histogram, filters, restoration, segmentation, edge_detection, morphology

app = FastAPI(
    title="Image Processing API",
    description="API for image processing operations extracted from the notebook",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # في production غيّرها لـ domain بتاعتك
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(point_ops.router,      prefix="/api/point-ops",    tags=["Point Operations"])
app.include_router(color.router,          prefix="/api/color",         tags=["Color Manipulation"])
app.include_router(histogram.router,      prefix="/api/histogram",     tags=["Histogram"])
app.include_router(filters.router,        prefix="/api/filters",       tags=["Filters"])
app.include_router(restoration.router,    prefix="/api/restoration",   tags=["Image Restoration"])
app.include_router(segmentation.router,   prefix="/api/segmentation",  tags=["Segmentation"])
app.include_router(edge_detection.router, prefix="/api/edge-detection",tags=["Edge Detection"])
app.include_router(morphology.router,     prefix="/api/morphology",    tags=["Morphology"])

@app.get("/")
def root():
    return {"message": "Image Processing API is running ✅", "docs": "/docs"}
