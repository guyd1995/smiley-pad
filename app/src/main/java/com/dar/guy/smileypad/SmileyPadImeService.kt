package com.dar.guy.smileypad

import android.app.Service
import android.content.Intent
import android.view.View
import android.os.Handler
import android.os.Message
import android.os.Messenger
import android.widget.Toast
import android.os.HandlerThread


class MessageHandler(handlerOwner: Service) : Handler() {
    var owner: Service = handlerOwner

    override fun handleMessage(msg : Message){
        Toast.makeText(owner, "text", Toast.LENGTH_LONG).show()
    }
}


class SmileyPadImeService : SmileyPadKeyboard(){

    private val messageHandler = MessageHandler(this)

    fun startCamera(view: View?){
        val intent = Intent(this, CameraActivity::class.java)
        intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        intent.putExtra("MESSENGER", Messenger(messageHandler))
        startActivity(intent)
    }
}
