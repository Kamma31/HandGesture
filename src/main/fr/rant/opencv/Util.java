package fr.rant.opencv;

import com.bulenkov.darcula.DarculaLaf;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import javax.swing.*;

public class Util {
    public static Mat getMatResource(final String filename) {
        return Imgcodecs.imread(getResource(filename));
    }

    public static Mat getMatResource(final String filename, final int imgcodecs) {
        return Imgcodecs.imread(getResource(filename), imgcodecs);
    }

    public static String getResource(final String filename) {
        return Util.class.getResource("/" + filename).getPath().substring(1);
    }

    public static void loadLibrairies() {
        try {
            UIManager.setLookAndFeel(new DarculaLaf());
        } catch (final UnsupportedLookAndFeelException ignored) {
        }
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

}
