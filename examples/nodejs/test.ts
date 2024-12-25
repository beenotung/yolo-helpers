import * as tf from '@tensorflow/tfjs-node'
import { detectPose, loadYoloModel } from '../../node'
import { resolve } from 'path'

async function main() {
  let file = '../browser/demo.jpg'
  let model = await loadYoloModel(
    '../browser/saved_models/yolo11n-pose_web_model',
  )
  let predictions = await detectPose({
    tf,
    file,
    model,
    maxOutputSize: 1,
    num_classes: 1,
    num_keypoints: 17,
  })
  console.log('predictions[0][0]:')
  console.dir(predictions[0][0], { depth: 0 })
}
main().catch(e => console.error(e))

// test different path formats
async function test() {
  // test path with or without `/model.json`
  async function test(url: string) {
    await loadYoloModel(url + '/model.json')
    await loadYoloModel(url)
  }

  // test local path
  await test('../browser/saved_models/yolo11n-pose_web_model')

  // test absolute path
  await test(
    'file://' + resolve('../browser/saved_models/yolo11n-pose_web_model'),
  )

  // test http path
  await test('http://localhost:8100/saved_models/yolo11n-pose_web_model')

  console.log('[pass] tested all variants of model url')
}
test().catch(e => console.error(e))
