package com.dar.guy.smileypad

import android.content.Intent
import android.view.View
import android.os.Handler
import android.os.Message
import android.os.Messenger
import android.util.Log
import android.widget.Toast
import org.json.JSONArray
import java.io.File


class MessageHandler(handlerOwner: SmileyPadImeService) : Handler() {
    val owner: SmileyPadImeService = handlerOwner
    var idxToEmoji: Array<String?>? = null

    init {
        val emojiJsonFile = File(Utils.assetFilePath(owner, "idx2emoji.json", true))
        val emojiJsonArr = JSONArray(emojiJsonFile.readText())
        idxToEmoji = arrayOfNulls<String>(emojiJsonArr.length())
        for (i in 0 until emojiJsonArr.length()) {
            idxToEmoji!![i] = emojiJsonArr.getString(i);
        }
    }

    override fun handleMessage(msg : Message){
        val emoji = msg.arg1
        val emojiText = idxToEmoji!![emoji]
        owner.curEmoji = emojiText
    }
}


class SmileyPadImeService : SmileyPadKeyboard(){

    private var messageHandler: MessageHandler? = null
    var curEmoji: String? = null
    override fun onCreate() {
        super.onCreate()
        messageHandler = MessageHandler(this)
    }

    override fun onWindowShown() {
        super.onWindowShown()
        if(curEmoji != null){
            val inputConnection = currentInputConnection
            inputConnection.commitText(curEmoji, 1)

        }

    }

    fun startCamera(view: View?){
        val intent = Intent(this, CameraActivity::class.java)
        intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        intent.putExtra("MESSENGER", Messenger(messageHandler))
        startActivity(intent)
    }
}
