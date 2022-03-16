## Welcome to GitHub Pages

You can use the [editor on GitHub](https://github.com/manginoja/455/edit/gh-pages/index.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)


```
### Summary
### Problem Setup

This model was made to be submitted to the Kaggle competition “Birds! Are they real?”.  The goal of this competition was to create a model that can classify bird species using the provided images and labels.  The model was then tested on a pre-defined dataset provided through the Kaggle competition, under the folder /test/.  The goal of this model was to provide the highest testing accuracy on this dataset.  

### Dataset

As this was for a Kaggle competition, this model used the dataset provided through the Kaggle site, under the /train/ folder. This folder was pre-organized into sub-folders, each containing a specific species of bird. This was in the format needed to process this data into a dataset using PyTorch’s dataset.ImageFolder function.  The only other preprocessing that was needed was splitting the dataset into an 80/20 train/test split.  

### Techniques

This model was primarily a modification of ResNet50, provided through the torchvision library. After testing nine separate combinations of learning rate (0.001, 0.01, 0.1) and decay (0.0005, 0.005, 0.05), the most promising combination was a learning rate of __ and decay of __, due to ___.

### Additional Info


### References

https://www.pluralsight.com/guides/introduction-to-resnet


For more details see [Basic writing and formatting syntax](https://docs.github.com/en/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/manginoja/455/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
