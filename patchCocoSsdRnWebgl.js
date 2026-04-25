import * as tf from '@tensorflow/tfjs';

/**
 * coco-ssd uses sync tf.image.nonMaxSuppression on the active backend; on rn-webgl that
 * blocks the UI thread. nonMaxSuppressionAsync pulls GPU data with await .data() and runs
 * CPU NMS — no backend switching (avoids lag from flipping cpu ↔ rn-webgl every frame).
 */
export function patchCocoSsdForRnWebgl(model) {
  if (!model || model.__cocoSsdRnPatched) {
    return;
  }

  model.infer = async function inferPatched(img, maxNumBoxes, minScore) {
    const batched = tf.tidy(() => {
      let input = img;
      if (!(input instanceof tf.Tensor)) {
        input = tf.browser.fromPixels(input);
      }
      return tf.expandDims(input);
    });

    const height = batched.shape[1];
    const width = batched.shape[2];
    const result = await this.model.executeAsync(batched);
    const scores = result[0].dataSync();
    const boxes = result[1].dataSync();
    const numBoxesOut = result[0].shape[1];
    const numClasses = result[0].shape[2];
    const boxLastDim = result[1].shape[3];

    batched.dispose();
    tf.dispose(result);

    const [maxScores, classes] = this.calculateMaxScores(scores, numBoxesOut, numClasses);

    const boxesTensor = tf.tensor2d(boxes, [numBoxesOut, boxLastDim]);
    let indexTensor;
    try {
      indexTensor = await tf.image.nonMaxSuppressionAsync(
        boxesTensor,
        maxScores,
        maxNumBoxes,
        minScore,
        minScore
      );
    } finally {
      boxesTensor.dispose();
    }

    const indexes = indexTensor.dataSync();
    indexTensor.dispose();

    return this.buildDetectedObjects(width, height, boxes, maxScores, indexes, classes);
  };

  Object.defineProperty(model, '__cocoSsdRnPatched', { value: true, enumerable: false });
}
