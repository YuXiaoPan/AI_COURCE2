package me.peyppicp.advance;

import com.vdurmont.emoji.Emoji;
import com.vdurmont.emoji.EmojiManager;
import org.apache.commons.io.Charsets;
import org.apache.commons.io.FileUtils;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * @author YuXiao Pan
 * @date 2017/12/6
 * @email yuxiao.pan@kikatech.com
 */
public class ReformatEmojiSample {

    public static void main(String[] args) throws IOException {

        File file = new File("emoji_sample.txt");
        List<String> samples = FileUtils.readLines(file, Charsets.UTF_8);
        List<String> newSamples = new ArrayList<>(samples.size());
        int i = 0;
        int total = samples.size() / 10000;
        for (String sample : samples) {
            char[] tempChars = new char[sample.length() + 1];
            for (Emoji emoji : EmojiManager.getAll()) {
                String unicode = emoji.getUnicode();
                if (sample.contains(unicode)) {
                    int firstIndex = sample.indexOf(unicode);
                    char[] chars = sample.toCharArray();
                    if (firstIndex >= 1) {
                        System.arraycopy(chars, 0, tempChars, 0, firstIndex);
                        tempChars[firstIndex] = ' ';
                        System.arraycopy(chars, firstIndex, tempChars, firstIndex + 1, tempChars.length - firstIndex - 1);
                    }
                    break;
                }
            }
            String s = new String(tempChars);
            newSamples.add(s);
            if (i % 10000 == 0) {
                System.out.println("Finish batch: " + i / 10000 + ", remain: " + (total - i / 10000));
            }
            i++;
        }


//        去重
        FileUtils.writeLines(new File("new_sample.txt"),
                "UTF-8",
                newSamples,
                "\n",
                false);
    }
}
