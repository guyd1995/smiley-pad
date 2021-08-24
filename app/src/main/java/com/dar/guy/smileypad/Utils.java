package com.dar.guy.smileypad;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.media.Image;
import android.util.Log;
import org.pytorch.Tensor;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.Arrays;

public class Utils {
    public static int INPUT_TENSOR_WIDTH = 224;
    public static int INPUT_TENSOR_HEIGHT = 224;

    public static float[] rgbToFloatArray(Image image) {
        Image.Plane[] planes = image.getPlanes();
        ByteBuffer rBuffer = planes[0].getBuffer();
        ByteBuffer gBuffer = planes[1].getBuffer();
        ByteBuffer bBuffer = planes[2].getBuffer();

        int rSize = rBuffer.remaining();
        int gSize = gBuffer.remaining();
        int bSize = bBuffer.remaining();

        byte[] rgbArr = new byte[rSize + gSize + bSize];
        rBuffer.get(rgbArr, 0, rSize);
        gBuffer.get(rgbArr, rSize, gSize);
        bBuffer.get(rgbArr, rSize + gSize, bSize);
        return byteArrayToNormalizedImage(rgbArr);
    }

    public static float[] flatImageToFloatArray(Image image) {
        ByteBuffer imgBuffer = image.getPlanes()[0].getBuffer();
        byte[] imgArr = new byte[imgBuffer.remaining()];
        imgBuffer.get(imgArr, 0, imgArr.length);

        return byteArrayToNormalizedImage(imgArr);
    }

    public static float[] yuvToFloatArray(Image image) {
        Image.Plane[] planes = image.getPlanes();
        ByteBuffer yBuffer = planes[0].getBuffer();
        ByteBuffer uBuffer = planes[1].getBuffer();
        ByteBuffer vBuffer = planes[2].getBuffer();

        int ySize = yBuffer.remaining();
        int uSize = uBuffer.remaining();
        int vSize = vBuffer.remaining();

        byte[] nv21 = new byte[ySize + uSize + vSize];
        yBuffer.get(nv21, 0, ySize);
        vBuffer.get(nv21, ySize, vSize);
        uBuffer.get(nv21, ySize + vSize, uSize);

        YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, image.getWidth(), image.getHeight(), null);
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        yuvImage.compressToJpeg(new Rect(0, 0, yuvImage.getWidth(), yuvImage.getHeight()), 75, out);

        byte[] imageBytes = out.toByteArray();
        return byteArrayToNormalizedImage(imageBytes);
    }

    public static FloatBuffer bitmapToFloatBuffer(Bitmap bitmap){
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();

        FloatBuffer inTensorBuffer = Tensor
                .allocateFloatBuffer(3 * width * height);
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                int colour = bitmap.getPixel(x, y);
                int red = Color.red(colour);
                int blue = Color.blue(colour);
                int green = Color.green(colour);
                inTensorBuffer.put(x + width * y, (float) red);
                inTensorBuffer.put(width * height + x + width * y, (float) green);
                inTensorBuffer.put(2 * width * height + x + width * y, (float) blue);
            }
        }
    return inTensorBuffer;
    }

    private static float[] byteArrayToNormalizedImage(byte[] imageBytes) {
        Bitmap bmp = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
        Bitmap resizedBmp = Bitmap.createScaledBitmap(bmp,
                INPUT_TENSOR_WIDTH, INPUT_TENSOR_HEIGHT, false);
        FloatBuffer inputBuffer = bitmapToFloatBuffer(resizedBmp);
        float[] input = new float[inputBuffer.remaining()];
        for(int i = 0; i < inputBuffer.remaining(); i++){
            input[i] = (inputBuffer.get(i) - 127.5F) / 128;
        }
        return input;
    }

    public static String assetFilePath(Context context, String assetName, boolean forceUpdate) {
        File file = new File(context.getFilesDir(), assetName);
        if (!forceUpdate && file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        } catch (IOException e) {
            Log.e("SmileyPadError", "Error process asset " + assetName + " to file path." +
                    e.toString());
        }
        return null;
    }

    public static int[] topK(float[] a, final int topk) {
        float values[] = new float[topk];
        Arrays.fill(values, -Float.MAX_VALUE);
        int ixs[] = new int[topk];
        Arrays.fill(ixs, -1);

        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < topk; j++) {
                if (a[i] > values[j]) {
                    for (int k = topk - 1; k >= j + 1; k--) {
                        values[k] = values[k - 1];
                        ixs[k] = ixs[k - 1];
                    }
                    values[j] = a[i];
                    ixs[j] = i;
                    break;
                }
            }
        }
        return ixs;
    }
}