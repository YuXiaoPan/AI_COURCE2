package me.peyppicp.advance2;

import com.vdurmont.emoji.Emoji;
import com.vdurmont.emoji.EmojiManager;
import com.vdurmont.emoji.EmojiParser;
import org.apache.commons.io.Charsets;
import org.apache.commons.io.FileUtils;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * @author YuXiao Pan
 * @date 2017/12/7
 * @email yuxiao.pan@kikatech.com
 */
public class ReformatEmojiSample2 {

    public static void main(String[] args) throws IOException {
        File file = new File("emoji_sample.txt");
        List<String> sampleLines = FileUtils.readLines(file, Charsets.UTF_8);
        List<String> emojiUnicodes = EmojiManager.getAll().parallelStream().map(Emoji::getUnicode).collect(Collectors.toList());
        List<String> temp = new ArrayList<>();
        List<String> errorLines = new ArrayList<>();
        int count = 0;
        int totalSize = sampleLines.size() / 1000;

//        按照emoji进行切分
        for (String line : sampleLines) {
            try {
                int emojiLength = 2;
                int currentEmojiIndex = 0;
                List<String> containedEmojis = EmojiParser.extractEmojis(line).parallelStream().distinct().collect(Collectors.toList());
                for (String emoji : containedEmojis) {
                    currentEmojiIndex = line.indexOf(emoji);
                    boolean flag = false;
                    if (currentEmojiIndex != -1) {
                        for (int i = currentEmojiIndex; i < line.length() - 1; i += emojiLength) {
                            if (EmojiManager.isEmoji(line.substring(currentEmojiIndex, currentEmojiIndex + emojiLength))) {
                                currentEmojiIndex += emojiLength;
                                flag = true;
                            }
                        }
                    }
                    if (flag) {
                        temp.add(line.substring(0, currentEmojiIndex).trim());
                        line = line.substring(currentEmojiIndex, line.length()).trim();
                    }
                }
            } catch (Exception e) {
                errorLines.add(line);
            }
            count++;
            if (count % 1000 == 0) {
                System.out.println("Remain: " + (totalSize - (count / 1000)));
            }
        }


        temp = temp.parallelStream().filter(s -> EmojiParser.extractEmojis(s).size() != s.length() / 2).distinct().collect(Collectors.toList());
        List<String> temp1 = new ArrayList<>();

//        添加空格
        for (String sample : temp) {
            String emoji = EmojiParser.extractEmojis(sample).get(0);
            int i = sample.indexOf(emoji);
            if (i >= 1) {
                if (i == emoji.length() - 1) {
                    continue;
                } else {
                    String head = sample.substring(0, i - 1);
                    String last = sample.substring(i, sample.length());
                    temp1.add(head + " " + last);
                }
            }
        }

//

        FileUtils.writeLines(new File("EmojiSample.txt"),
                "UTF-8",
                temp1,
                "\n",
                false);
    }
}
