import * as tf from '@tensorflow/tfjs'
import {
  detectPose,
  loadYoloModel,
  drawBox,
  detectPoseSync,
  detectSegment,
  detectBox,
  combineMask,
  hasOverlap,
  classifyImage,
} from '../../browser'

declare let app: HTMLElement
declare let image: HTMLImageElement
declare let canvas: HTMLCanvasElement

let context = canvas.getContext('2d')!

let image_src = 'demo.jpg'
let maxOutputSize = 1

let warm_up_rounds = 10
warm_up_rounds = 0

image_src = 'example.png'
maxOutputSize = 20

let scoreThreshold = 0.2
let iouThreshold = 0.1

async function main_classify() {
  let num_classes = 1000

  console.log('loading model...')
  let model = await loadYoloModel('saved_models/yolo11n-cls_web_model')
  console.log({ model })

  let input_height = model.inputs[0].shape![1]
  let input_width = model.inputs[0].shape![2]

  console.log('loading image...')
  await new Promise(resolve => {
    image.src = image_src
    image.onload = resolve
  })

  console.log('warm up...')
  for (let i = 0; i < warm_up_rounds; i++) {
    console.time('classifyImage')
    await classifyImage({
      model,
      pixels: image,
      num_classes,
    })
    console.timeEnd('classifyImage')
  }

  console.log('predicting...')
  let predictions = await classifyImage({
    model,
    pixels: image,
    num_classes,
  })
  console.log({ predictions })

  image.style.width = `${input_width}px`
  image.style.height = `${input_height}px`
  let rect = image.getBoundingClientRect()
  canvas.width = rect.width
  canvas.height = rect.height
  canvas.style.width = rect.width + 'px'
  canvas.style.height = rect.height + 'px'

  console.log('drawing...')
  context.clearRect(0, 0, canvas.width, canvas.height)

  for (let prediction of predictions) {
    let list = document.createElement('ol')
    prediction.all_confidences
      .map((confidence, index) => {
        let item = document.createElement('li')
        item.textContent = `${confidence.toFixed(2)} (${index}:${
          model.class_names?.[index]
        })`
        return { item, confidence }
      })
      .sort((a, b) => b.confidence - a.confidence)
      .forEach(({ item }) => {
        list.appendChild(item)
      })
    document.body.appendChild(list)
  }

  console.log('done')
}

async function main_box() {
  console.log('loading model...')
  let model = await loadYoloModel('saved_models/yolo11n_web_model_640')
  // let model = await loadYoloModel('saved_models/yolo11n_web_model')
  console.log({ model })

  let input_height = model.inputs[0].shape![1]
  let input_width = model.inputs[0].shape![2]

  console.log('loading image...')
  await new Promise(resolve => {
    image.src = image_src
    image.onload = resolve
  })

  console.log('warm up...')
  for (let i = 0; i < warm_up_rounds; i++) {
    console.time('detectBox')
    await detectBox({
      tf,
      model,
      pixels: image,
      maxOutputSize,
      num_classes: model.class_names!.length,
      scoreThreshold,
      iouThreshold,
    })
    console.timeEnd('detectBox')
  }

  console.log('predicting...')
  let predictions = await detectBox({
    tf,
    model,
    pixels: image,
    maxOutputSize,
    num_classes: model.class_names!.length,
    scoreThreshold,
    iouThreshold,
  })
  console.log({ predictions })

  image.style.width = `${input_width}px`
  image.style.height = `${input_height}px`
  let rect = image.getBoundingClientRect()
  canvas.width = rect.width
  canvas.height = rect.height
  canvas.style.width = rect.width + 'px'
  canvas.style.height = rect.height + 'px'

  console.log('drawing...')
  context.clearRect(0, 0, canvas.width, canvas.height)
  for (let prediction of predictions) {
    for (let box of prediction) {
      let confidence = box.confidence.toFixed(2)
      let index = box.class_index
      let name = model.class_names?.[index] || '?'
      drawBox({
        context,
        x: box.x,
        y: box.y,
        width: box.width,
        height: box.height,
        borderColor: 'red',
        label: {
          text: `${confidence} (${index}:${name})`,
          fontColor: 'cyan',
          backgroundColor: '#0005',
        },
      })
    }
  }

  console.log('done')
}

async function main_pose() {
  console.log('loading model...')
  let model = await loadYoloModel('saved_models/yolo11n-pose_web_model')
  console.log({ model })

  let input_height = model.inputs[0].shape![1]
  let input_width = model.inputs[0].shape![2]

  console.log('loading image...')
  await new Promise(resolve => {
    image.src = 'demo.jpg'
    image.onload = resolve
  })

  console.log('warm up...')
  for (let i = 0; i < warm_up_rounds; i++) {
    console.time('detectPose')
    await detectPose({
      tf,
      model,
      pixels: image,
      maxOutputSize,
      num_classes: model.class_names!.length,
      num_keypoints: model.keypoints!,
      visibility: model.visibility!,
    })
    console.timeEnd('detectPose')
  }

  console.log('predicting...')
  let predictions = await detectPose({
    tf,
    model,
    pixels: image,
    maxOutputSize,
    num_classes: model.class_names!.length,
    num_keypoints: model.keypoints!,
    visibility: model.visibility!,
  })
  console.log({ predictions })

  image.style.width = `${input_width}px`
  image.style.height = `${input_height}px`
  let rect = image.getBoundingClientRect()
  canvas.width = rect.width
  canvas.height = rect.height
  canvas.style.width = rect.width + 'px'
  canvas.style.height = rect.height + 'px'

  console.log('drawing...')
  context.clearRect(0, 0, canvas.width, canvas.height)
  for (let prediction of predictions) {
    for (let box of prediction) {
      let confidence = box.confidence.toFixed(2)
      let index = box.class_index
      let name = model.class_names?.[index] || '?'
      drawBox({
        context,
        x: box.x,
        y: box.y,
        width: box.width,
        height: box.height,
        borderColor: 'red',
        label: {
          text: `${confidence} (${index}:${name})`,
          fontColor: 'cyan',
          backgroundColor: '#0005',
        },
      })
      for (let keypoint of box.keypoints) {
        drawBox({
          context,
          x: keypoint.x,
          y: keypoint.y,
          width: 3,
          height: 3,
          borderColor: 'red',
          label: {
            text: keypoint.visibility.toFixed(2),
            fontColor: 'cyan',
            backgroundColor: '#0005',
          },
        })
      }
    }
  }

  console.log('done')
}

async function main_segment() {
  console.log('loading model...')
  // let model = await loadYoloModel('saved_models/yolo11n-seg_web_model_640')
  // let model = await loadYoloModel('saved_models/yolo11n-seg_web_model')
  let model = await loadYoloModel('saved_models/tongue-seg-v1_web_model')
  console.log({ model })

  let input_height = model.inputs[0].shape![1]
  let input_width = model.inputs[0].shape![2]

  console.log('loading image...')
  await new Promise(resolve => {
    image.src = image_src
    image.onload = resolve
  })

  console.log('warm up...')
  let warm_up_rounds = 10
  warm_up_rounds = 0
  for (let i = 0; i < warm_up_rounds; i++) {
    console.time('detectSegment')
    await detectSegment({
      tf,
      model,
      pixels: image,
      maxOutputSize,
      num_classes: model.class_names!.length,
      scoreThreshold,
      iouThreshold,
    })
    console.timeEnd('detectSegment')
  }

  console.log('predicting...')
  let predictions = await detectSegment({
    tf,
    model,
    pixels: image,
    maxOutputSize,
    num_classes: model.class_names!.length,
    scoreThreshold,
    iouThreshold,
  })
  console.log({ predictions })

  image.style.width = `${input_width}px`
  image.style.height = `${input_height}px`
  let rect = image.getBoundingClientRect()
  canvas.width = rect.width
  canvas.height = rect.height
  canvas.style.width = rect.width + 'px'
  canvas.style.height = rect.height + 'px'

  console.log('drawing...')
  context.clearRect(0, 0, canvas.width, canvas.height)
  for (let prediction of predictions) {
    for (let box of prediction.bounding_boxes) {
      let confidence = box.confidence.toFixed(2)
      let index = box.class_index
      let name = model.class_names?.[index] || '?'
      drawBox({
        context,
        x: box.x,
        y: box.y,
        width: box.width,
        height: box.height,
        borderColor: 'red',
        label: {
          text: `${confidence} (${index}:${name})`,
          fontColor: 'cyan',
          backgroundColor: '#0005',
        },
      })

      let boxRect = {
        left: box.x - box.width / 2,
        top: box.y - box.height / 2,
        right: box.x + box.width / 2,
        bottom: box.y + box.height / 2,
      }

      let mask = combineMask(box, prediction.masks)
      let H = mask.length
      let W = mask[0].length

      for (let h = 0; h < H; h++) {
        for (let w = 0; w < W; w++) {
          let width = canvas.width / W
          let height = canvas.height / H

          let left = (w / W) * canvas.width
          let top = (h / H) * canvas.height

          let right = left + width
          let bottom = top + height

          let maskRect = { left, top, right, bottom }

          if (!hasOverlap(boxRect, maskRect)) {
            continue
          }

          let v = mask[h][w]
          context.fillStyle = `rgba(0,255,0,${v})`

          context.fillRect(left, top, width, height)
        }
      }
    }
  }

  console.log('done')
}

let main = main_segment

main().catch(e => console.error(e))
