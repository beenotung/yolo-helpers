# yolo-helpers

Helper functions to use models converted from YOLO in browser and Node.js.

[![npm Package Version](https://img.shields.io/npm/v/yolo-helpers)](https://www.npmjs.com/package/yolo-helpers)
[![Minified Package Size](https://img.shields.io/bundlephobia/min/yolo-helpers)](https://bundlephobia.com/package/yolo-helpers)
[![Minified and Gzipped Package Size](https://img.shields.io/bundlephobia/minzip/yolo-helpers)](https://bundlephobia.com/package/yolo-helpers)

## Features

- Support for YOLO models:
  - [Pose](https://docs.ultralytics.com/tasks/pose/)
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

  // Detect poses in the image
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

## Typescript Signature

### Common Types

```typescript
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

type Keypoint = {
  /** x of keypoint in px */
  x: number
  /** y of keypoint in px */
  y: number
  /** confidence of keypoint */
  visibility: number
}

type BoundingBoxWithKeypoints = BoundingBox & {
  keypoints: Keypoint[]
}

/**
 * output shape: [batch, box]
 *
 * Array of batches, each containing array of detected bounding boxes
 * */
type PoseResult = BoundingBoxWithKeypoints[][]

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
}
```

### Browser Functions

```typescript
// Load YOLO model in browser
function loadYoloModel(
  /**
   * Can be with or without `/model.json`.
   * Examples:
   * - "./saved_model/yolo11n-pose_web_model/model.json"
   * - "./saved_model/yolo11n-pose_web_model"
   * - "http://localhost:8100/saved_models/yolo11n-pose_web_model/model.json"
   * - "https://domain.net/saved_models/yolo11n-pose_web_model"
   * - "indexeddb://yolo11n-pose_web_model"
   */
  modelUrl: string,
): Promise<tf.InferenceModel>

// Detect poses in browser
function detectPose(args: DetectPoseArgs & ImageInput): Promise<PoseResult>

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

### Node.js Functions

```typescript
// Load YOLO model in Node.js
function loadYoloModel(
  /**
   * Can be with or without `/model.json`.
   * Examples:
   * - "./saved_model/yolo11n-pose_web_model/model.json"
   * - "./saved_model/yolo11n-pose_web_model"
   * - "file://path/to/model.json"
   * - "http://localhost:8100/saved_models/yolo11n-pose_web_model"
   * - "https://domain.net/saved_models/yolo11n-pose_web_model/model.json"
   */
  modelPath: string,
): Promise<tf.InferenceModel>

// Detect poses in Node.js
function detectPose(args: DetectPoseArgs & ImageInput): Promise<PoseResult>

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

## License

This project is licensed with [BSD-2-Clause](./LICENSE)

This is free, libre, and open-source software. It comes down to four essential freedoms [[ref]](https://seirdy.one/2021/01/27/whatsapp-and-the-domestication-of-users.html#fnref:2):

- The freedom to run the program as you wish, for any purpose
- The freedom to study how the program works, and change it so it does your computing as you wish
- The freedom to redistribute copies so you can help others
- The freedom to distribute copies of your modified versions to others
