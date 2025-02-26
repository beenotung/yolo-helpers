import * as tf from '@tensorflow/tfjs'
import {
  detectPose,
  loadYoloModel,
  drawBox,
  detectPoseSync,
} from '../../browser'

declare let app: HTMLElement
declare let image: HTMLImageElement
declare let canvas: HTMLCanvasElement

let context = canvas.getContext('2d')!

let input_width = 640
let input_height = 640
let num_boxes =
  (input_width / 8) * (input_height / 8) +
  (input_width / 16) * (input_height / 16) +
  (input_width / 32) * (input_height / 32)
let num_classes = 1
let num_keypoints = 17
let length = 4 + num_classes + num_keypoints * 3

async function main() {
  console.log('loading model...')
  let model = await loadYoloModel('saved_models/yolo11n-pose_web_model')
  console.log({ model })

  console.log('loading image...')
  await new Promise(resolve => {
    image.src = 'demo.jpg'
    image.onload = resolve
  })

  console.log('warm up...')
  let warm_up_rounds = 10
  warm_up_rounds = 0
  for (let i = 0; i < warm_up_rounds; i++) {
    console.time('detectPose')
    await detectPose({
      tf,
      model,
      pixels: image,
      maxOutputSize: 1,
      num_classes,
      num_keypoints,
    })
    console.timeEnd('detectPose')
  }

  console.log('predicting...')
  let predictions = await detectPose({
    tf,
    model,
    pixels: image,
    maxOutputSize: 1,
    num_classes,
    num_keypoints,
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
      drawBox({
        context,
        x: box.x,
        y: box.y,
        width: box.width,
        height: box.height,
        borderColor: 'red',
        label: {
          text: box.confidence.toFixed(2),
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

main().catch(e => console.error(e))
