package me.peyppicp.advance2;

import com.google.common.base.Preconditions;
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
 * @date 2017/12/8
 * @email yuxiao.pan@kikatech.com
 */
public class BuildSampleLabel2 {

    public static void main(String[] args) throws IOException {
        File file = new File("ReEnforcementEmojiSample.txt");
        List<String> samples = FileUtils.readLines(file, Charsets.UTF_8);
        WordToIndex wordToIndex = new WordToIndex("ReEnforcementEmojiSample.txt");
        ArrayList<String> labels = new ArrayList<>();
        for (String sample : samples) {
            List<String> emojis = EmojiParser.extractEmojis(sample)
                    .parallelStream().distinct().collect(Collectors.toList());
            StringBuilder sb = new StringBuilder();
            if (emojis.size() == 0) {
                int index = wordToIndex.getIndex(WordToIndex.STOP);
                labels.add(String.valueOf(index));
                continue;
            }
            for (String emoji : emojis) {
                int index = wordToIndex.getIndex(emoji);
//                if (index == -1) {
//                    index = wordToIndex.getIndex(WordToIndex.UNKNOWN);
//                    sb.append(WordToIndex.UNKNOWN).append("-").append(index).append(",");
//                }else{
//                    sb.append(emoji).append("-").append(index).append(",");
//                }
                sb.append(index).append(",");
            }
            sb.deleteCharAt(sb.length() - 1);
            labels.add(sb.toString());
        }
        Preconditions.checkArgument(samples.size() == labels.size());
        FileUtils.writeLines(new File("ReEnforcementEmojiSampleLabels.txt"),
                "UTF-8",
                labels,
                "\n",
                false);
    }
}
