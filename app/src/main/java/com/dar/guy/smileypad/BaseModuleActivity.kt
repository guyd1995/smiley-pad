package com.dar.guy.smileypad

import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity


open class BaseModuleActivity : AppCompatActivity() {
    protected var mBackgroundThread: HandlerThread? = null
    protected var mBackgroundHandler: Handler? = null
    protected var mUIHandler: Handler? = null
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        mUIHandler = Handler(mainLooper)
    }

    protected fun startBackgroundThread() {
        mBackgroundThread = HandlerThread("ModuleActivity")
        mBackgroundThread!!.start()
        mBackgroundHandler = Handler(mBackgroundThread!!.looper)
    }

    override fun onDestroy() {
        stopBackgroundThread()
        super.onDestroy()
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

}