# Multiclass Image Classification with FastAI

In this project, I built a **multiclass image classification system** using the `fastai` library. 

---

## Dataset Collection

I used the `duckduckgo_search` package to scrape training images for each category.

This was very interesting, I was aware that automation tools for image/data scraping on the internet existed.
However up until now I had never attempted this process.

```python
categories = 'airplane', 'automobile', 'bird', 'cat', 'dog'
```

All images were resized to 400px to ensure consistent preprocessing and model input.


##  Data Loading with DataBlock

FastAI's `DataBlock` API allowed me to define the data pipeline. The data was split 80/20 for training and validation using `RandomSplitter`:

```python
dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
).dataloaders(path)
```

This gave me a clean and efficient interface for image loading and augmentation.

---

## Training with ResNet18

I fine-tuned a pretrained ResNet18 model using FastAI's `vision_learner` and `fine_tune` utilities:

```python
learn = vision_learner(dls, resnet18, metrics=accuracy)
learn.fine_tune(3)
```

Using the GPU Frozen dev container, This only took a few seconds per epoch.



## Model Evaluation

### Confusion Matrix
The confusion matrix helped identify where the model confused certain classes. For example, most misclassifications were between bird/dog or airplane/automobile.

### t-SNE Visualization
I used `TSNE` to reduce the high-dimensional output vectors into 2D for visualization:

```python
from sklearn.manifold import TSNE
```

The resulting plot showed clear, separated clusters, indicating the model was effective in multiclass image classification.



## What I Learned

- How to scrape and clean image datasets from scratch
- How to structure and use the FastAI `DataBlock` API
- The power of transfer learning with pretrained models
- How to interpret confusion matrices and t-SNE projections

This assignment was a great entry point into implementing simple image classification models with API's, 
and analysing the models effectiviness with t-SNE and confusion matrices.

---

---

## References
- [FastAI Documentation](https://docs.fast.ai/)
- [DuckDuckGo Image Search](https://pypi.org/project/duckduckgo-search/)
- [t-SNE Explantion](https://www.youtube.com/watch?v=RJVL80Gg3lA)

