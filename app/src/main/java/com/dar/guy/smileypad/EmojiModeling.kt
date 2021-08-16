package com.dar.guy.smileypad

import android.content.Context
import androidx.annotation.Nullable
import androidx.annotation.WorkerThread
import androidx.camera.core.ImageProxy
import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.File;
import java.nio.FloatBuffer;
import java.util.LinkedList;
import java.util.Locale;
import java.util.Queue;

import android.os.SystemClock
import android.util.Log
import java.lang.Exception

private val idx2Emoji = arrayOf("Neutral", "Happy", "Sad")

private const val INPUT_TENSOR_WIDTH = 224
private const val INPUT_TENSOR_HEIGHT = 224
private const val TOP_K = 3


class EmojiModeling {
        private val mModuleAssetName: String = "face_model.pt"
        private var mModule: Module? = null
        private var mInputTensorBuffer: FloatBuffer? = null
        private var mInputTensor: Tensor? = null
        private var mAnalyzeImageErrorState = false

        class AnalysisResult(
            private val topNClassNames: Array<String?>, private val topNScores: FloatArray,
            private val moduleForwardDuration: Long, private val analysisDuration: Long
        )

        @WorkerThread
        @Nullable
        fun analyzeImage(context: Context, image: ImageProxy, rotationDegrees: Int): AnalysisResult? {
            return if (mAnalyzeImageErrorState) {
                null
            } else try {
                if (mModule == null) {
                    val moduleFileAbsoluteFilePath = File(
                        Utils.assetFilePath(context, mModuleAssetName)
                    ).absolutePath
                    mModule = Module.load(moduleFileAbsoluteFilePath)
                    mInputTensorBuffer =
                        Tensor.allocateFloatBuffer(3 * INPUT_TENSOR_WIDTH * INPUT_TENSOR_HEIGHT)
                    mInputTensor = Tensor.fromBlob(
                        mInputTensorBuffer,
                        longArrayOf(1, 3, INPUT_TENSOR_HEIGHT.toLong(), INPUT_TENSOR_WIDTH.toLong())
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
                val topKClassNames = arrayOfNulls<String>(TOP_K)
                val topKScores = FloatArray(TOP_K)
                for (i in 0 until TOP_K) {
                    val ix = ixs[i]
                    topKClassNames[i] = idx2Emoji[ix]
                    topKScores[i] = scores[ix]
                }
                val analysisDuration = SystemClock.elapsedRealtime() - startTime
                AnalysisResult(topKClassNames, topKScores, moduleForwardDuration, analysisDuration)
            } catch (e: Exception) {
                Log.e("SmileyPadError", "Error during image analysis", e)
                mAnalyzeImageErrorState = true
                null
            }
        }

    fun destroy(){
        mModule?.destroy()
    }

}