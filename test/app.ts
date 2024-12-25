import * as tf from '@tensorflow/tfjs'
import {
  checkYoloPoseOutput,
  decodeYoloPose,
  predictYoloPose,
} from '../yolo-pose-utils'

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
  let model = await tf.loadGraphModel(
    'saved_models/yolo11n-pose_web_model/model.json',
  )
  console.log({ model })

  console.log('loading image...')
  await new Promise(resolve => {
    image.src = 'demo.jpg'
    image.onload = resolve
  })

  console.log('predicting...')
  let predictions = await predictYoloPose({
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
      drawRect({
        x: box.x,
        y: box.y,
        width: box.width,
        height: box.height,
        color: 'red',
        text: box.cls_score.toFixed(2),
      })
      for (let keypoint of box.keypoints) {
        drawRect({
          x: keypoint.x,
          y: keypoint.y,
          width: 3,
          height: 3,
          color: 'cyan',
          text: keypoint.visibility.toFixed(2),
        })
      }
    }
  }

  console.log('done')
}

main().catch(e => console.error(e))

function drawRect(input: {
  x: number
  y: number
  width: number
  height: number
  color: string
  text: string
  lineWidth?: number
}) {
  let lineWidth = input.lineWidth ?? 5

  let left = input.x - input.width / 2
  let top = input.y - input.height / 2

  context.lineWidth = lineWidth
  context.strokeStyle = input.color
  context.strokeRect(left, top, input.width, input.height)

  context.font = '12px Arial'
  context.fillText(input.text, left, top > 5 ? top - 5 : top + lineWidth)
}
