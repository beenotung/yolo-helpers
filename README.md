# yolo-helpers

Helper functions to use models converted from YOLO in browser and Node.js.

[![npm Package Version](https://img.shields.io/npm/v/yolo-helpers)](https://www.npmjs.com/package/yolo-helpers)
[![Minified Package Size](https://img.shields.io/bundlephobia/min/yolo-helpers)](https://bundlephobia.com/package/yolo-helpers)
[![Minified and Gzipped Package Size](https://img.shields.io/bundlephobia/minzip/yolo-helpers)](https://bundlephobia.com/package/yolo-helpers)

## Features

- Support for YOLO models:
  - [Image Classification](https://docs.ultralytics.com/tasks/classify/) (classify images into one of the given classes)
  - [Object Detection](https://docs.ultralytics.com/tasks/detect/) (detect objects and location of bounding boxes)
  - [Keypoint Detection](https://docs.ultralytics.com/tasks/pose/) (detect objects and location of keypoints)
  - [Instance Segmentation](https://docs.ultralytics.com/tasks/segment/) (detect objects and generate segmentation masks)
- Typescript support
- Isomorphic package: works in Node.js and browsers

## Installation

```bash
npm install yolo-helpers
```

You can also install `yolo-helpers` with [pnpm](https://pnpm.io/), [yarn](https://yarnpkg.com/), or [slnpm](https://github.com/beenotung/slnpm)

## Usage Examples

For complete examples, see [examples/browser/app.ts](./examples/browser/app.ts) and [examples/nodejs/test.ts](./examples/nodejs/test.ts)

### Browser

```typescript
import * as tf from '@tensorflow/tfjs'
import { detectPose, loadYoloModel } from 'yolo-helpers/dist/browser'

async function main() {
  // Load the YOLO model
  const model = await loadYoloModel('url/to/yolo11n-pose_web_model')

  // Get image element
  const image = document.querySelector('img')!

  // Detect poses in the image element
  const predictions = await detectPose({
    tf,
    model,
    pixels: image,
    maxOutputSize: 1,
    num_classes: 1,
    num_keypoints: 17,
  })

  // predictions[0] contains array of detected poses with bounding boxes and keypoints
  console.log(predictions[0])
}
```

### Node.js

```typescript
import * as tf from '@tensorflow/tfjs-node'
import { detectPose, loadYoloModel } from 'yolo-helpers'

async function main() {
  // Load the YOLO model
  const model = await loadYoloModel('path/to/yolo11n-pose_web_model')

  // Detect poses in an image file
  const predictions = await detectPose({
    tf,
    model,
    file: 'path/to/image.jpg',
    maxOutputSize: 1,
    num_classes: 1,
    num_keypoints: 17,
  })

  // predictions[0] contains array of detected poses with bounding boxes and keypoints
  console.log(predictions[0])
}
```

### Remark on Inference Speed

Sync version of `detectPose` is available as `detectPoseSync` but it is not recommended.

The sync version is slower than the async version even when running in the browser/nodejs without other concurrent tasks.

If you want to speed up the inference, export the model with smaller `imgsz` (e.g. `imgsz=[256,320]` for 256px height, 320px width).

It's similar case for `detectSegment`, `detectBox` and `classifyImage`.

## Typescript Signature

### Model Functions

```typescript
function loadYoloModel(modelUrl: string): Promise<
  tf.InferenceModel & {
    class_names?: string[]
  }
>
```

The `modelUrl` can be with or without `/model.json`.

<details>
<summary> Example `modelUrl` for browser: </summary>

- `./saved_model/yolo11n-pose_web_model/model.json`
- `./saved_model/yolo11n-pose_web_model`
- `http://localhost:8100/saved_models/yolo11n-pose_web_model/model.json`
- `https://domain.net/saved_models/yolo11n-pose_web_model`
- `indexeddb://yolo11n-pose_web_model`

</details>

<details>
<summary> Example `modelUrl` for node.js: </summary>

- `./saved_model/yolo11n-pose_web_model/model.json`
- `./saved_model/yolo11n-pose_web_model`
- `file://path/to/model.json`
- `http://localhost:8100/saved_models/yolo11n-pose_web_model`
- `https://domain.net/saved_models/yolo11n-pose_web_model/model.json`

</details>

### Detection Functions and Types

<details>
<summary> classifyImage() </summary>

```typescript
/**
 * image features:
 *   - confidence of all classes
 *   - highest confidence, class_index
 *
 * The confidence are already normalized between 0 to 1, and sum up to 1.
 */
function classifyImage(args: ClassifyArgs): Promise<ClassifyResult>

/**
 * output shape: [batch]
 *
 * Array of batches, each containing array of confidence for each classes
 * */
type ClassifyResult = ImageResult[]

type ImageResult = {
  /** class index with highest confidence */
  class_index: number
  /** confidence of the class with highest confidence */
  confidence: number
  /** confidence of all classes */
  all_confidences: number[]
}

type ClassifyArgs = {
  model: tf.InferenceModel
  /** used for image resize when necessary, auto inferred from model shape */
  input_shape?: {
    width: number
    height: number
  }
  /** e.g. `1` for single class */
  num_classes: number
} & ImageInput
```

</details>

<details>
<summary> detectBox() </summary>

```typescript
/**
 * box features:
 *   - x, y, width, height
 *   - highest confidence, class_index
 *
 * The x, y, width, height are in pixel unit, NOT normalized in the range of [0, 1].
 * The the pixel units are scaled to the input_shape.
 *
 * The confidence are already normalized between 0 to 1.
 */
function detectBox(args: DetectBoxArgs): Promise<BoxResult>

/**
 * output shape: [batch, box]
 *
 * Array of batches, each containing array of detected bounding boxes
 * */
type BoxResult = BoundingBox[][]

type BoundingBox = {
  /** center x of bounding box in px */
  x: number
  /** center y of bounding box in px */
  y: number
  /** width of bounding box in px */
  width: number
  /** height of bounding box in px */
  height: number
  /** class index with highest confidence */
  class_index: number
  /** confidence of the class with highest confidence */
  confidence: number
  /** confidence of all classes */
  all_confidences: number[]
}

type DetectBoxArgs = {
  model: tf.InferenceModel
  /** used for image resize when necessary, auto inferred from model shape */
  input_shape?: {
    width: number
    height: number
  }
  /**
   * tensorflow runtime:
   * - browser: `import * as tf from '@tensorflow/tfjs'`
   * - nodejs: `import * as tf from '@tensorflow/tfjs-node'`
   */
  tf: typeof tf_type
  /** e.g. `1` for single class */
  num_classes: number
  /**
   * Number of boxes to return using non-max suppression.
   * If not provided, all boxes will be returned
   *
   * e.g. `1` for only selecting the bounding box with highest confidence.
   */
  maxOutputSize?: number
  /**
   * the threshold for deciding whether boxes overlap too much with respect to IOU.
   *
   * default: `0.5`
   */
  iouThreshold?: number
  /**
   * the threshold for deciding whether a box is a valid detection.
   *
   * default: `-Infinity`
   */
  scoreThreshold?: number
} & ImageInput
```

</details>

<details>
<summary> detectPose() </summary>

```typescript
/**
 * box features:
 *   - x, y, width, height
 *   - highest confidence, class_index
 *   - keypoints
 *
 * keypoint features:
 *   - x, y, visibility
 *
 * The x, y, width, height are in pixel unit, NOT normalized in the range of [0, 1].
 * The the pixel units are scaled to the input_shape.
 *
 * The confidence are already normalized between 0 to 1.
 */
function detectPose(args: DetectPoseArgs & ImageInput): Promise<PoseResult>

/**
 * output shape: [batch, box]
 *
 * Array of batches, each containing array of detected bounding boxes
 * */
type PoseResult = BoundingBoxWithKeypoints[][]

type BoundingBoxWithKeypoints = BoundingBox & {
  keypoints: Keypoint[]
}

type Keypoint = {
  /** x of keypoint in px */
  x: number
  /** y of keypoint in px */
  y: number
  /** confidence of keypoint */
  visibility: number
}

type DetectPoseArgs = {
  model: tf.InferenceModel
  /** used for image resize when necessary, auto inferred from model shape */
  input_shape?: {
    width: number
    height: number
  }
  /**
   * tensorflow runtime:
   * - browser: `import * as tf from '@tensorflow/tfjs'`
   * - nodejs: `import * as tf from '@tensorflow/tfjs-node'`
   */
  tf: typeof tf_type
  /** e.g. `1` for single class */
  num_classes: number
  /** e.g. `17` for 17 keypoints */
  num_keypoints: number
  /**
   * Number of boxes to return using non-max suppression.
   * If not provided, all boxes will be returned
   *
   * e.g. `1` for only selecting the bounding box with highest confidence.
   */
  maxOutputSize: number
  /**
   * the threshold for deciding whether boxes overlap too much with respect to IOU.
   *
   * default: `0.5`
   */
  iouThreshold?: number
  /**
   * the threshold for deciding whether a box is a valid detection.
   *
   * default: `-Infinity`
   */
  scoreThreshold?: number
} & ImageInput
```

</details>

<details>
<summary> detectSegment() </summary>

```typescript
/**
 * boxes features:
 *   - x, y, width, height
 *   - highest confidence, class_index
 *   - mask coefficients for each channel
 *
 * mask features:
 * - [height, width, channel]: 0 for background, 1 for object
 *
 * The x, y, width, height are in pixel unit, NOT normalized in the range of [0, 1].
 * The the pixel units are scaled to the input_shape.
 *
 * The confidence are already normalized between 0 to 1.
 */
function detectSegment(args: DetectSegmentArgs): Promise<SegmentResult>

/**
 * output shape: [batch, box]
 *
 * Array of batches, each containing array of detected bounding boxes with masks coefficients and masks
 * */
type SegmentResult = {
  bounding_boxes: BoundingBoxWithMaskCoefficients[]
  /** e.g. [mask_height, mask_width, 32] for 32 channels of masks */
  masks: Mask[]
}[]

type BoundingBoxWithMaskCoefficients = BoundingBox & {
  /** 32 coefficients of mask */
  mask_coefficients: number[]
}

/** [height, width, num_channels] -> 0 for background, 1 for object */
type Mask = number[][]

type DetectSegmentArgs = {
  model: tf.InferenceModel
  /** used for image resize when necessary, auto inferred from model shape */
  input_shape?: {
    width: number
    height: number
  }
  /**
   * tensorflow runtime:
   * - browser: `import * as tf from '@tensorflow/tfjs'`
   * - nodejs: `import * as tf from '@tensorflow/tfjs-node'`
   */
  tf: typeof tf_type
  /** e.g. `1` for single class */
  num_classes: number
  /**
   * Number of channels in segmentation mask
   * default: `32`
   */
  num_channels?: number
  /**
   * Number of boxes to return using non-max suppression.
   * If not provided, all boxes will be returned
   *
   * e.g. `1` for only selecting the bounding box with highest confidence.
   */
  maxOutputSize?: number
  /**
   * the threshold for deciding whether boxes overlap too much with respect to IOU.
   *
   * default: `0.5`
   */
  iouThreshold?: number
  /**
   * the threshold for deciding whether a box is a valid detection.
   *
   * default: `-Infinity`
   */
  scoreThreshold?: number
} & ImageInput
```

</details>

<details>
<summary> ImageInput type for browser </summary>

```typescript
type ImageInput =
  | {
      pixels:
        | PixelData
        | ImageData
        | HTMLImageElement
        | HTMLCanvasElement
        | HTMLVideoElement
        | ImageBitmap
    }
  | {
      /**
       * input shape: [height, width, channels] or [batch, height, width, channels]
       *
       * the pixel values should be in the range of [0, 255]
       */
      tensor: tf.Tensor
    }
```

</details>

<details>
<summary> ImageInput type for node.js </summary>

```typescript
type ImageInput =
  | {
      /** path to image file */
      file: string
    }
  | {
      /**
       * input shape: [height, width, channels] or [batch, height, width, channels]
       *
       * the pixel values should be in the range of [0, 255]
       */
      tensor: tf.Tensor
    }
```

</details>

### Helper Functions for drawing

<details>
<summary> drawBox() </summary>

```typescript
function drawBox(args: {
  /** canvas context to draw on */
  context: CanvasRenderingContext2D

  /** x-axis of the center of the box, in pixel unit */
  x: number

  /** y-axis of the center of the box, in pixel unit */
  y: number

  /** width of the box, in pixel unit */
  width: number

  /** height of the box, in pixel unit */
  height: number

  /** color of the border of the box, default is `red` */
  borderColor?: string

  /** line width of the box, in pixel unit, default is 5px */
  lineWidth?: number

  /** label of the box, e.g. class name, confidence score, etc. */
  label?: {
    text: string
    /** color of the text label, default is `'white'` */
    fontColor?: string
    /** background color of the text label, default is `'transparent'` */
    backgroundColor?: string
    /** font style of the text label, default is `'normal 900 14px Arial, sans-serif'` */
    font?: string
  }
}): void
```

</details>

## License

This project is licensed with [BSD-2-Clause](./LICENSE)

This is free, libre, and open-source software. It comes down to four essential freedoms [[ref]](https://seirdy.one/2021/01/27/whatsapp-and-the-domestication-of-users.html#fnref:2):

- The freedom to run the program as you wish, for any purpose
- The freedom to study how the program works, and change it so it does your computing as you wish
- The freedom to redistribute copies so you can help others
- The freedom to distribute copies of your modified versions to others
