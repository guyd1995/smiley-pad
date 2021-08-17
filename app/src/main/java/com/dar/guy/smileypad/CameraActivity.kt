package com.dar.guy.smileypad

import android.Manifest
import android.content.pm.PackageManager
import android.os.*
import android.util.Size
import android.view.TextureView
import android.widget.Toast
import androidx.annotation.UiThread
import androidx.camera.core.*
import androidx.camera.core.Preview.OnPreviewOutputUpdateListener
import androidx.camera.core.Preview.PreviewOutput
import androidx.core.app.ActivityCompat
import android.view.View
import android.view.ViewStub
import android.text.TextUtils;
import android.util.Log;
import android.widget.Button
import android.widget.TextView;
import androidx.appcompat.app.AppCompatActivity


class CameraActivity :  AppCompatActivity() {
    protected var mBackgroundThread: HandlerThread? = null
    protected var mBackgroundHandler: Handler? = null
    protected var mUIHandler: Handler? = null
    private var mLastAnalysisResultTime: Long = 0
    private var mEmojiModeling: EmojiModeling? = null
    private var messageHandler: MessageHandler? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        messageHandler = intent.extras?.get("MESSENGER") as MessageHandler
        mUIHandler = Handler(mainLooper)
        mEmojiModeling = EmojiModeling()

        setContentView(R.layout.activity_camera)
        findViewById<Button>(R.id.send_emoji).setOnClickListener {
            sendResultAndDie()
        }

        startBackgroundThread()
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(
                this,
                PERMISSIONS,
                REQUEST_CODE_CAMERA_PERMISSION
            )
        } else {
            setupCameraX()
        }
    }

    fun sendResultAndDie(){
        val msg = Message.obtain()
        messageHandler?.sendMessage(msg)
        finish()
    }

    protected fun startBackgroundThread() {
        mBackgroundThread = HandlerThread("ModuleActivity")
        mBackgroundThread!!.start()
        mBackgroundHandler = Handler(mBackgroundThread!!.looper)
    }

    protected fun stopBackgroundThread() {
        mBackgroundThread!!.quitSafely()
        try {
            mBackgroundThread!!.join()
            mBackgroundThread = null
            mBackgroundHandler = null
        } catch (e: InterruptedException) {
            Toast.makeText(this, "Error on stopping background thread",
                Toast.LENGTH_LONG).show()
        }
    }

    override fun onDestroy() {
        mEmojiModeling?.destroy()
        stopBackgroundThread()
        super.onDestroy()
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults: IntArray
    ) {
        if (requestCode == REQUEST_CODE_CAMERA_PERMISSION) {
            if (grantResults[0] == PackageManager.PERMISSION_DENIED) {
                Toast.makeText(
                    this,
                    "You can't use image classification example without granting CAMERA permission",
                    Toast.LENGTH_LONG
                )
                    .show()
                finish()
            } else {
                setupCameraX()
            }
        }
    }

    private fun setupCameraX() {
        val textureView = getCameraPreviewTextureView()
        val previewConfig = PreviewConfig.Builder().setLensFacing(CameraX.LensFacing.FRONT).build()
        val preview = Preview(previewConfig)
        preview.onPreviewOutputUpdateListener =
            OnPreviewOutputUpdateListener { output: PreviewOutput ->
                textureView?.surfaceTexture = output.surfaceTexture
            }
        val imageAnalysisConfig = ImageAnalysisConfig.Builder()
            .setLensFacing(CameraX.LensFacing.FRONT)
            .setTargetResolution(Size(224, 224))
            .setCallbackHandler(mBackgroundHandler!!)
            .setImageReaderMode(ImageAnalysis.ImageReaderMode.ACQUIRE_LATEST_IMAGE)
            .build()
        val imageAnalysis = ImageAnalysis(imageAnalysisConfig)
        imageAnalysis.analyzer =
            ImageAnalysis.Analyzer { image: ImageProxy?, rotationDegrees: Int ->
                if (SystemClock.elapsedRealtime() - mLastAnalysisResultTime < 500) {
                    return@Analyzer
                }
                val result = mEmojiModeling!!.analyzeImage(this, image!!, rotationDegrees)
                if (result != null) {
                    mLastAnalysisResultTime = SystemClock.elapsedRealtime()
                    runOnUiThread { applyToUiAnalyzeImageResult(result) }
                }
            }
        CameraX.bindToLifecycle(this, preview, imageAnalysis)
    }

    protected fun getCameraPreviewTextureView(): TextureView? {
        return (findViewById<View>(R.id.texture_view_stub) as ViewStub)
            .inflate()
            .findViewById(R.id.texture_view)
    }

    @UiThread
    protected fun applyToUiAnalyzeImageResult(result: EmojiModeling.AnalysisResult?){

    }

    companion object {
        private const val REQUEST_CODE_CAMERA_PERMISSION = 200
        private val PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }
}