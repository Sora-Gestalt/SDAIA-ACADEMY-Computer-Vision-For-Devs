# Damaged Apple Detection System

**Author:** NAWAF ABDULLAH BINTALEB  
**Program:** SDAIA Academy Computer Vision  
**Cohort Dates:** Spring 2026  
**Academy:** [SDAIA Academy](https://github.com/SDAIAAcademy)

---

## Executive Summary

The **Damaged Apple Detection System** is a computer vision and decoupled web service designed to automate quality control in agricultural supply chains and produce packaging facilities. Built using **YOLO**, **FastAPI**, and **Streamlit**, the application isolates defects and damages on apples in real-time. By decoupling model inference into a dedicated cloud REST API from the user-facing web dashboard, the system guarantees low-latency processing, modular scalability, and seamless integration into automated sorting lines.

---

## Architectural Rationale & Design Choices

### Section 1: Setup and Environment Configuration
> **Rationale:** Standardizing package dependencies across `ultralytics`, `opencv-python-headless`, `fastapi`, and `streamlit` ensures runtime compatibility across training, API backend, and frontend UI environments. Isolating requirements early prevents dependency conflicts between heavy neural network libraries and web frameworks.

### Section 2: Dataset Acquisition & Annotation Setup
> **Rationale:** Sourcing annotated visual data via Roboflow in YOLO format structures object classes and bounding box coordinates into standardized YAML configs. Setting up explicit train/validation splits upfront ensures unbiased performance evaluation across fresh and damaged produce categories.

### Section 3: Model Training & Hyperparameter Tuning
> **Rationale:** Utilizing a lightweight YOLO detection backbone trained on GPU acceleration (such as Google Colab T4) balances inference speed with detection accuracy. Adjusting image size ($640\times640$), batch size, and epoch counts optimizes mean Average Precision (mAP50) for identifying subtle physical flaws like bruises, rot, or skin cuts.

### Section 4: Quantitative Validation & Metric Evaluation
> **Rationale:** Evaluating the model against Precision, Recall, and mAP metrics isolates specific failure modes (e.g., misclassifying natural apple stem marks as damage). Independent validation testing ensures reliable performance threshold setting before pushing model weights into production REST endpoints.

### Section 5: FastAPI Microservice & Model Decoupling
> **Rationale:** Decoupling model execution into a dedicated FastAPI endpoint prevents bundling heavy weight files directly inside client application interfaces. Asynchronous file uploads via REST endpoints enable horizontal scaling of GPU backend workers independently of frontend web traffic.

### Section 6: Streamlit UI & Interactive Image Processing
> **Rationale:** Deploying a Streamlit web application allows field inspectors and quality control managers to capture live camera footage or upload fruit batch images seamlessly. Integrating OpenCV rendering on top of API response payloads dynamically renders high-visibility bounding boxes, defect labels, and confidence metrics.

### Section 7: Confidence Threshold Tuning & User Controls
> **Rationale:** Providing dynamic confidence sliders in the user interface empowers operators to adjust detection sensitivity on the fly based on specific lighting conditions or grading strictness in different processing facilities.

### Section 8: Quality Control & Care/Action Knowledge Base
> **Rationale:** Mapping detected defect classes directly to immediate action guidelines (e.g., routing heavily damaged apples to juice processing vs. minor flaws to discounted sales) automates decision-making for non-specialized staff.

### Section 9: Integration & End-to-End System Verification
> **Rationale:** End-to-end integration testing across sample test images ensures robust communication between the Streamlit client and FastAPI model backend. Programmatically handling connection timeouts or empty detection payloads guarantees system stability during live usage.

---

## Repository Structure

```text
.
├── main.py              # FastAPI server handling model inference & API endpoints
├── app.py               # Streamlit web application frontend
├── best.pt              # Trained YOLO model weights for apple damage detection
├── train_plant_detector.ipynb # Note Book in google colab that the model has trained on
├── requirements.txt     # Python dependencies & package requirements
├── sample_images/       # Test image directory for validation & evaluation
└── README.md            # Technical documentation & architectural rationale
```


## Application Showcase Video



https://github.com/user-attachments/assets/cda77e42-8867-4637-be99-53999c07f36a










