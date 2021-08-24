package com.dar.guy.smileypad

import android.content.Context
import androidx.annotation.Nullable
import androidx.annotation.WorkerThread
import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;

import java.io.File;
import java.nio.FloatBuffer;

import android.os.SystemClock
import org.pytorch.LiteModuleLoader

private val idx2Emoji: Array<Int> = arrayOf(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)

private const val TOP_K = 3


class EmojiModeling {
        private val mModuleAssetName: String = "face_model.ptl"
        private var mModule: Module? = null
        private var mInputTensorBuffer: FloatBuffer? = null
        private var mInputTensor: Tensor? = null
        private var mAnalyzeImageErrorState = false

        class AnalysisResult(
            val value: Int,
            private val topNClassNames: Array<Int?>, private val topNScores: FloatArray,
            private val moduleForwardDuration: Long, private val analysisDuration: Long
        )

        @WorkerThread
        @Nullable
        fun analyzeImage(context: Context, rgbArray: FloatArray,
                         rotationDegrees: Int): AnalysisResult? {
            return if (mAnalyzeImageErrorState) {
                null
            } else {

                if (mModule == null) {
                    val moduleFileAbsoluteFilePath = File(
                        Utils.assetFilePath(context, mModuleAssetName, true)
                    ).absolutePath

                    val nPixels = Utils.INPUT_TENSOR_WIDTH * Utils.INPUT_TENSOR_HEIGHT
                    mModule = LiteModuleLoader.load(moduleFileAbsoluteFilePath)
                    mInputTensorBuffer =
                        Tensor.allocateFloatBuffer(3 * nPixels)
                    mInputTensorBuffer!!.put(rgbArray)
                    mInputTensor = Tensor.fromBlob(
                        mInputTensorBuffer,
                        longArrayOf(1, 3, Utils.INPUT_TENSOR_HEIGHT.toLong(),
                            Utils.INPUT_TENSOR_WIDTH.toLong())
                    )
                }
                val startTime = SystemClock.elapsedRealtime()

//                TensorImageUtils.imageYUV420CenterCropToFloatBuffer(
//                    image.image, rotationDegrees,
//                    INPUT_TENSOR_WIDTH, INPUT_TENSOR_HEIGHT,
//                    TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
//                    TensorImageUtils.TORCHVISION_NORM_STD_RGB,
//                    mInputTensorBuffer, 0
//                )
                val moduleForwardStartTime = SystemClock.elapsedRealtime()
                val outputTensor = mModule!!.forward(IValue.from(mInputTensor)).toTensor()

                val moduleForwardDuration = SystemClock.elapsedRealtime() - moduleForwardStartTime
                val scores = outputTensor.dataAsFloatArray

                val ixs = Utils.topK(scores, TOP_K)
                val topKClassNames = arrayOfNulls<Int>(TOP_K)
                val topKScores = FloatArray(TOP_K)
                for (i in 0 until TOP_K) {
                    val ix = ixs[i]
                    topKClassNames[i] = idx2Emoji[ix]
                    topKScores[i] = scores[ix]
                }
                val value = topKClassNames[0]
                val analysisDuration = SystemClock.elapsedRealtime() - startTime
                AnalysisResult(value!!, topKClassNames, topKScores, moduleForwardDuration, analysisDuration)
            }
//            catch (e: Exception) {
//                Log.e("SmileyPadError", "Error during image analysis", e)
//                mAnalyzeImageErrorState = true
//                null
//            }
        }

    fun destroy(){
        mModule?.destroy()
    }

}