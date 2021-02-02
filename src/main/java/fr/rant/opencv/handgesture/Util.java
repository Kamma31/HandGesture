package fr.rant.opencv.handgesture;

import com.bulenkov.darcula.DarculaLaf;
import nu.pattern.OpenCV;

import javax.swing.*;

public class Util {
    public static void loadLibrairies() {
        try {
            UIManager.setLookAndFeel(new DarculaLaf());
        } catch (final UnsupportedLookAndFeelException ignored) {
        }
        OpenCV.loadLocally();
    }

}
