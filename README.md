# fb_marketplace_reco
Facebook Marketplace is a platform for buying and selling products on Facebook. This project involves training a multimodal deep neural network model that predicts the category of a product based on its image and text description. The trained model is then dockerized and served using FastAPI.

The multimodel model learning is a scaled down implementation of a part of a larger system that Facebook developed to generate product recommendation rankings for buyers. See [here](https://engineering.fb.com/2018/10/02/ml-applications/under-the-hood-facebook-marketplace-powered-by-artificial-intelligence/) for more details.

## Datasets:
- A tabular dataset consisting of product details is assumed to be available from an RDS system
- An image dataset consisting of product images is assumed to be available in an S3 bucket

## Image classification

### Image cleaning and transformation
- clean_images.py provides sklearn transformations for image processing; this is used for HOG-SVM based image classification. This is a traditional approach compared to the final CNN based approach.
- prepare_image_data.py cleans and prepares images for CNN based image classification.

### Model training
- prod_pred.py uses Histogram Oriented Gradient - Support Vector Machine (HOG-SVM) approach to classify the product images. This is not intended to be used in the final model.
- image_classification.ipynb trains the image classification model using RESNET-50 (transfer learning). Only the last three layers of RESNET-50 are trained, which yileds a feature vector. Then the model adds a classification module on top of the feature vector.
<!-- ![System](/visuals/image_cnn_training.png)
*Training curve of the transfer learning with RESNET-50.* -->

## Text classification

### Text cleaning and preprocessing
- clean_tabular.py provides sklearn pipelines to do basic cleaning of the tabular dataset
- word2vec_util.TextCorpusProcess implements nltk based text preprocessing. It implements stop words removal, lemmatizing and stemming.

### Model training
- word2vec.ipynb: trains the word2vec embeddings (natural language modeling) to be used in downstream text classification. It uses the skip-gram model with negative sampling.
![System](/visuals/w2v.png)
*Word2vec embedding of the first 300 words plotted using t-SNE*

- text_classification.ipynb trains the text classification model as a CNN model operating in the word2vec embedding space.
<!-- ![System](/visuals/text_cnn_training.png)
*Training curve of the text classification model.* -->

## Multimodal image-text classification

### Model training
- combined_classification.ipynb trains a multimodal image-text classification model using the already trained image and text models. The final classification layers are removed from both the image and text CNN models and the respective feature vectors concatenated to form a combined embedding vector. This is followed by a classification layer to make the product prediction. The image and text models are frozen during training.
<!-- ![System](/visuals/combined_training.png) -->
<!-- *Training curve of the multimodal classification model.* -->

## Serving
The trained models are dockerized and served using FastAPI. This is implemented as a separate repository which is included as the submodule api. See api/fb_mk_api.py for the implementation. Three endpoints are provided each for image, text and combined:
- 0.0.0.0:8008/image
- 0.0.0.0:8008/text
- 0.0.0.0:8008/combined
The 0.0.0.0 ip address is a placeholder.

The json response is represented as follows:
res = JSONResponse(status_code=200, content={
        'pred': pred, 'classes': classes}), 
where pred is the predicted product category and classes is the ranked categories with the most probable appearing first.

## Outcome
- The combined model does perform better than the individual image or text models.
- The dataset used is too small and this is reflected in the low accuracy scores of the models and the large gap between the training and validation curves.

## Future investigations
- Use BERT instead of training a word2vec embedding model from scratch
- Do not freeze the word2vec model when training the combined model.
- Data augmentation to enhance the dataset size.





