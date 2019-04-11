package org.neuroph.samples.adalineDigits;

import org.neuroph.core.data.DataSetRow;

/**
 * Training data for digit recognition adaline example.
 */
public class DigitData {

    public final static int CHAR_WIDTH = 5;
    public final static int CHAR_HEIGHT = 7;

    public static String[][] DIGITS = {
           {" OOO ",
            "O   O",
            "O   O",
            "O   O",
            "O   O",
            "O   O",
            " OOO "},

           {"  O  ",
            " OO  ",
            "O O  ",
            "  O  ",
            "  O  ",
            "  O  ",
            "  O  "},

           {" OOO ",
            "O   O",
            "    O",
            "   O ",
            "  O  ",
            " O   ",
            "OOOOO"},

           {" OOO ",
            "O   O",
            "    O",
            " OOO ",
            "    O",
            "O   O",
            " OOO "},

           {"   O ",
            "  OO ",
            " O O ",
            "O  O ",
            "OOOOO",
            "   O ",
            "   O "},

           {"OOOOO",
            "O    ",
            "O    ",
            "OOOO ",
            "    O",
            "O   O",
            " OOO "},

           {" OOO ",
            "O   O",
            "O    ",
            "OOOO ",
            "O   O",
            "O   O",
            " OOO "},

           {"OOOOO",
            "    O",
            "    O",
            "   O ",
            "  O  ",
            " O   ",
            "O    "},

           {" OOO ",
            "O   O",
            "O   O",
            " OOO ",
            "O   O",
            "O   O",
            " OOO "},

           {" OOO ",
            "O   O",
            "O   O",
            " OOOO",
            "    O",
            "O   O",
            " OOO "}};

    public static DataSetRow convertImageIntoData(String[] image) {

        DataSetRow dataSetRow = new DataSetRow(DigitData.CHAR_HEIGHT * DigitData.CHAR_WIDTH);

        double[] array = new double[DigitData.CHAR_WIDTH * DigitData.CHAR_HEIGHT];

        for (int row = 0; row < DigitData.CHAR_HEIGHT; row++) {
            for (int column = 0; column < DigitData.CHAR_WIDTH; column++) {
                int index = (row * DigitData.CHAR_WIDTH) + column;
                char ch = image[row].charAt(column);
                array[index] = (ch == 'O' ? 1 : -1);
            }
        }
        dataSetRow.setInput(array);
        return dataSetRow;
    }

    public static String[] convertDataIntoImage(double[] data) {

        String[] image = new String[data.length / DigitData.CHAR_WIDTH];
        String row = "";

        for (int i = 0; i < data.length; i++) {
            if (data[i] == 1) {
                row += "O";
            } else {
                row += " ";
            }
            if (row.length() % 5 == 0 && row.length() != 0) {
                image[i / 5] = row;
                row = "";
            }
        }
        return image;
    }
}
