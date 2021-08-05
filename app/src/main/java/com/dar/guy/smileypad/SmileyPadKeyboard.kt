package com.dar.guy.smileypad

import android.inputmethodservice.InputMethodService
import android.view.View
import android.inputmethodservice.Keyboard
import android.inputmethodservice.KeyboardView
import android.text.TextUtils
import android.view.KeyEvent


open class SmileyPadKeyboard : InputMethodService(), KeyboardView.OnKeyboardActionListener{
    private var keyboardView: KeyboardView? = null
    private var keyboard: Keyboard? = null

    private var caps = false

    override fun onCreateInputView(): View {
        keyboardView = layoutInflater.inflate(R.layout.keyboard_view, null) as KeyboardView
        keyboard = Keyboard(this, R.xml.keys_layout)
        keyboardView!!.keyboard = keyboard
        keyboardView!!.setOnKeyboardActionListener(this)
        return keyboardView!!
    }


    override fun onPress(i: Int) {

    }

    override fun onRelease(i: Int) {

    }

    override fun onKey(primaryCode: Int, keyCodes: IntArray) {
        val inputConnection = currentInputConnection
        if (inputConnection != null) {
            when (primaryCode) {
                Keyboard.KEYCODE_DELETE -> {
                    val selectedText = inputConnection.getSelectedText(0)

                    if (TextUtils.isEmpty(selectedText)) {
                        inputConnection.deleteSurroundingText(1, 0)
                    } else {
                        inputConnection.commitText("", 1)
                    }
                    caps = !caps
                    keyboard!!.setShifted(caps)
                    keyboardView!!.invalidateAllKeys()
                }
                Keyboard.KEYCODE_SHIFT -> {
                    caps = !caps
                    keyboard!!.setShifted(caps)
                    keyboardView!!.invalidateAllKeys()
                }
                Keyboard.KEYCODE_DONE -> inputConnection.sendKeyEvent(
                    KeyEvent(
                        KeyEvent.ACTION_DOWN,
                        KeyEvent.KEYCODE_ENTER
                    )
                )
                else -> {
                    var code = primaryCode.toChar()
                    if (Character.isLetter(code) && caps) {
                        code = Character.toUpperCase(code)
                    }
                    inputConnection.commitText(code.toString(), 1)
                }
            }
        }

    }

    override fun onText(charSequence: CharSequence) {

    }

    override fun swipeLeft() {

    }

    override fun swipeRight() {

    }

    override fun swipeDown() {

    }

    override fun swipeUp() {

    }
}