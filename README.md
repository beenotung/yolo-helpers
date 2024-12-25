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

### Browser

```typescript
import * as tf from '@tensorflow/tfjs'
import { detectPose, loadYoloModel } from 'yolo-helpers/browser'

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
import { detectPose, loadYoloModel } from 'yolo-helpers/node'

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
export type BoundingBox = {
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

export type Keypoint = {
  /** x of keypoint in px */
  x: number
  /** y of keypoint in px */
  y: number
  /** confidence of keypoint */
  visibility: number
}

export type BoundingBoxWithKeypoints = BoundingBox & {
  keypoints: Keypoint[]
}

/**
 * output shape: [batch, box]
 *
 * Array of batches, each containing array of detected bounding boxes
 * */
export type PoseResult = BoundingBoxWithKeypoints[][]
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
type DetectPoseArgs = {
  model: tf.InferenceModel
  input_shape?: {
    width: number
    height: number
  }
  tf: typeof tf_type
  num_classes: number
  num_keypoints: number
  maxOutputSize: number
  iouThreshold?: number
  scoreThreshold?: number
} & (
  | { pixels: Parameters<typeof tf.browser.fromPixels>[0] } // HTML elements like Image, Canvas, Video
  | { tensor: tf.Tensor }
)

function detectPose(args: DetectPoseArgs): Promise<PoseResult>
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
type DetectPoseArgs = {
  model: tf.InferenceModel
  input_shape?: {
    width: number
    height: number
  }
  tf: typeof tf_type
  num_classes: number
  num_keypoints: number
  maxOutputSize: number
  iouThreshold?: number
  scoreThreshold?: number
} & (
  | { file: string } // File path to image
  | { tensor: tf.Tensor }
)

function detectPose(args: DetectPoseArgs): Promise<PoseResult>
```

## License

This project is licensed with [BSD-2-Clause](./LICENSE)

This is free, libre, and open-source software. It comes down to four essential freedoms [[ref]](https://seirdy.one/2021/01/27/whatsapp-and-the-domestication-of-users.html#fnref:2):

- The freedom to run the program as you wish, for any purpose
- The freedom to study how the program works, and change it so it does your computing as you wish
- The freedom to redistribute copies so you can help others
- The freedom to distribute copies of your modified versions to others
