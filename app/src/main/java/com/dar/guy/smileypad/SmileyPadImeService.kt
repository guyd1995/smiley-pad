package com.dar.guy.smileypad

import android.content.Intent
import android.view.View

class SmileyPadImeService : SmileyPadKeyboard(){
    fun startCamera(view: View?){
        val intent = Intent(this, CameraActivity::class.java)
        intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        startActivity(intent)
    }
}