package me.peyppicp.advance2;

import com.google.common.base.Preconditions;
import com.vdurmont.emoji.EmojiParser;
import org.apache.commons.io.Charsets;
import org.apache.commons.io.FileUtils;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * @author YuXiao Pan
 * @date 2017/12/8
 * @email yuxiao.pan@kikatech.com
 */
public class RemoveEmojis {

    public static void main(String[] args) throws IOException {
        File file = new File("ReEnforcementEmojiSample.txt");
        List<String> samples = FileUtils.readLines(file, Charsets.UTF_8);
        ArrayList<String> result = new ArrayList<>();
        for (String sample : samples) {
            String s = EmojiParser.removeAllEmojis(sample);
            result.add(s.trim());
        }

        Preconditions.checkArgument(samples.size() == result.size());
        FileUtils.writeLines(new File("ReEnforcementEmojiSampleWithoutEmoji.txt"),
                "UTF-8",
                result,
                "\n",
                false);
    }
}
