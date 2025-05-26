const rcnn = [
  {
    title: "RCNN",
    link: "https://ritvik19.medium.com/papers-explained-14-rcnn-ede4db2de0ab",
    date: "November 2013",
    description:
      "Uses selective search for region proposals, CNNs for feature extraction, SVM for classification followed by box offset regression.",
    tags: ["Object Detection", "Convolutional Neural Networks"],
  },
  {
    title: "Fast RCNN",
    link: "https://ritvik19.medium.com/papers-explained-15-fast-rcnn-28c1792dcee0",
    date: "April 2015",
    description:
      "Processes entire image through CNN, employs RoI Pooling to extract feature vectors from ROIs, followed by classification and BBox regression.",
    tags: ["Object Detection", "Convolutional Neural Networks"],
  },
  {
    title: "Faster RCNN",
    link: "https://ritvik19.medium.com/papers-explained-16-faster-rcnn-a7b874ffacd9",
    date: "June 2015",
    description:
      "A region proposal network (RPN) and a Fast R-CNN detector, collaboratively predict object regions by sharing convolutional features.",
    tags: ["Object Detection", "Convolutional Neural Networks"],
  },
  {
    title: "Mask RCNN",
    link: "https://ritvik19.medium.com/papers-explained-17-mask-rcnn-82c64bea5261",
    date: "March 2017",
    description:
      "Extends Faster R-CNN to solve instance segmentation tasks, by adding a branch for predicting an object mask in parallel with the existing branch.",
    tags: ["Object Detection", "Convolutional Neural Networks"],
  },
  {
    title: "Cascade RCNN",
    link: "https://ritvik19.medium.com/papers-explained-77-cascade-rcnn-720b161d86e4",
    date: "December 2017",
    description:
      "Proposes a multi-stage approach where detectors are trained with progressively higher IoU thresholds, improving selectivity against false positives.",
    tags: ["Object Detection", "Convolutional Neural Networks"],
  },
];
