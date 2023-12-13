import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Gradient-free deep learning: Experimental results",
    page_icon="🧪",
    layout="centered",
    initial_sidebar_state="expanded",
)


st.title("Experimental results")
st.header("Dataset")
st.image("./assets/coco_demo.png", caption="COCO vehicles dataset example: https://cocodataset.org/", use_column_width=True)
st.markdown(
    """The COCO (Common Objects in Context) Vehicles dataset is a subset of the larger COCO dataset, which is widely used in computer vision for object detection, segmentation, and image recognition tasks. This specific subset focuses on vehicles, encompassing a variety of vehicle types such as cars, trucks, buses, motorcycles, and bicycles.

Key features of the COCO Vehicles dataset include:

1. **Rich Annotations**: Each vehicle in the dataset is meticulously annotated, providing not just labels but also precise object boundaries. This facilitates tasks like instance segmentation and object detection.

2. **Diverse Contexts**: Vehicles are captured in various settings, from urban streets to rural areas, in different weather conditions and times of day, providing a realistic and challenging environment for computer vision models.

3. **Large Scale**: The dataset contains a significant number of images, each with one or more vehicle instances, making it suitable for training robust machine learning models.

4. **Variety of Vehicles**: It covers a wide range of vehicles, offering a comprehensive resource for tasks that require specific vehicle recognition or more general vehicle-related studies.

5. **Integration with COCO**: Being a part of the larger COCO dataset, it benefits from the same structure and standards, making it easy to integrate with other COCO subsets for multi-object recognition tasks.

The COCO Vehicles dataset is particularly valuable for researchers and practitioners working on autonomous driving, traffic monitoring systems, and general vehicle recognition in urban environments.
"""
)


st.header("Test loss")
df = pd.read_excel("./data/results.xlsx")
df.set_index('Method', inplace=True)
st.write(df)

st.header("Test accuracy")
df = pd.read_excel("./data/accuracy.xlsx")
df.set_index('Method', inplace=True)
st.write(df)
