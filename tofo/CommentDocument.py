from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tofo.processing import ToFoTextPreprocessor


def comment_document_parser(json_data, text_processor: ToFoTextPreprocessor):
    res_dict = dict()
    root_doc = None
    for elem in json_data:
        if "your comment has been removed" in elem['text'] or elem['text'] == '[removed]':
            continue
        doc = CommentDocument(elem['parent'] if elem['parent'] != 0 else None, elem['id'], elem['author'], elem['text'], text_processor)
        if doc.parent_id is None:
            root_doc = doc
        else:
            if doc.parent_id not in res_dict or len(doc.phrases_by_sentence) == 0 or len(doc.words_by_sentence) == 0:
                continue
            res_dict[doc.parent_id].children.append(doc)
        res_dict[doc.doc_id] = doc

    return root_doc


class CommentDocument(object):
    def __init__(self, parent_id: str, doc_id: str, author: str, text: str, text_processor: ToFoTextPreprocessor):
        self.parent_id = parent_id
        self.doc_id = doc_id
        self.author = author
        self.text = text_processor.preprocessing(text)

        self.children = list()
        self.sentences = text_processor.doc_to_sentence(self.text)

        self.words = text_processor.doc_to_words(self.text)
        self.words_by_sentence = [y for y in [text_processor.doc_to_words(x) for x in self.sentences] if len(y) > 0]

        self.phrases = text_processor.doc_to_phrases(self.text)
        self.phrases_by_sentence = [y for y in [text_processor.doc_to_phrases(x) for x in self.sentences] if len(y) > 0]

        self.bow = set(self.words)
        self.bow_by_sentence = [set(x) for x in self.words_by_sentence]

        self.bop = set(self.phrases)
        self.bop_by_sentence = [set(x) for x in self.phrases_by_sentence]

        self.wvec_by_sentence = None  # binary feature vector
        self.pvec_by_sentence = None  # binary feature vector

        self.wcluster_by_sentence = None  # clusters of bag-of-words sentences
        self.pcluster_by_sentence = None  # clusters of bag-of-phrases sentences

        self.wcluster_by_sentence_connection = None  # cluster connection of bag-of-words sentences
        self.pcluster_by_sentence_connection = None  # clusters connection of bag-of-phrases sentences

        self.wtop_by_sentence = None
        self.ptop_by_sentence = None


if __name__ == '__main__':
    preprocessor = ToFoTextPreprocessor()
    input_text = 'I am only speaking for the US, because of the decision in *Gideon vs. Wainwright* ' \
                 'that guarantees the right to an attorney. \n\nPlease note that my wording is "should", ' \
                 'because I am not so much interested in the practicalities of how long they should have to do ' \
                 'it, or setting up the infrastructure for it, but the general concept of it. I imagine it would' \
                 ' be like a doctor where you have rotations in different fields, though.\n\nI do not think this ' \
                 'system would be a panacea for the myriad ills facing our criminal justice system, but being ' \
                 'able to see these people as people will help prosecutors make a more informed and wholly ' \
                 'considered opinion before charging a defendant.\n\nProsecutors in the US have a massive ' \
                 'amount of discretion in how to apply the law. Think of someone like [Angela Corey](https://' \
                 'www.thenation.com/article/is-angela-corey-the-cruelest-prosecutor-in-america/). Having to ' \
                 'defend someone is the best way to foster empathy and to get a person like Corey to reconsider ' \
                 'her attitude towards "wrongdoers". There are numerous examples of other prosecutors who think ' \
                 'like her, or are so caught up in getting their cases that they use [irresponsible](https://ww' \
                 'w.washingtonpost.com/news/the-watch/wp/2016/09/07/white-house-science-council-bite-mark-mat' \
                 'ching-is-junk-science/?utm_term=.59622111532f) means to get convictions. If those convictions' \
                 ' are overturned or challenged, I have rarely (if) ever seen a DA\'s office admit a mistake. ' \
                 'They double down and argue that they were right; I don\'t know if they even try to take a ' \
                 'fresh look at the case.\n\nIn regards to conflicts of interest, just like any other case, ' \
                 'prosecutors or PD\'s would recuse themselves. \n\nI recognize that some positions are elected, ' \
                 'so I would add a stipulation requiring all candidates to have served a rotation as a PD before ' \
                 'being eligible to run.\n\nWhen it comes to judges, many of them were prosecutors, so this would' \
                 ' affect many of them as well, and affect their sentencing patterns.\n\nMy focus is on public' \
                 ' defenders because they are tasked with defending the destitute, the ones with long rap sheets' \
                 ' and no money. They have to talk to the defendant, their friends and families, get to know them' \
                 ' and fight for their interests while working in an [underfunded](https://www.themarshallproject.' \
                 'org/2016/09/07/when-the-money-runs-out-for-public-defense-what-happens-next#.2bhg69uPJ), ' \
                 '[understaffed](https://www.themarshallproject.org/2016/09/09/what-happens-when-there-s-only-' \
                 'one-public-defender?utm_medium=social&utm_campaign=sprout&utm_source=facebook#.G9gzNI7Fe), ' \
                 'and [undervalued](https://www.themarshallproject.org/2016/09/08/when-real-estate-and-tax-' \
                 'lawyers-are-forced-to-do-a-public-defender-s-job?utm_medium=social&utm_campaign=sprout&utm_' \
                 'source=facebook#.amg9jzWOw) system.\n\nAlrighty then, CMV!\n\n\nedit: To clarify what my s' \
                 'ystem would look like: \n\nPeople who want to be considered for prosecutor positions would ' \
                 'first have to complete a rotation as a PD, then go on to become a prosecutor. They would not ' \
                 'have to return to being a PD. I imagine it being like a doctor where you do rotations in ' \
                 'different specialties before you go into your field of choice. A podiatrist would not have ' \
                 'to serve as an OB/GYN after their rotation, and a prosecutor would not have to serve as a PD ' \
                 'after their rotation.\n\nAlso, I gave a delta to /u/stanthemanchan for suggesting a shadowing/' \
                 'assisting system as a good alternative to putting the future prosecutors in charge of the cases.'

    cdoc = CommentDocument(None, None, None, input_text, preprocessor)

    from tofo.processing import BOWTopicDetector, BOPTopicDetector, BOWFlowConstructor, BOPFlowConstructor

    BOWtopicDetector = BOWTopicDetector()
    BOPtopicDetector = BOPTopicDetector()

    BOWtopicDetector.fit(cdoc)
    BOPtopicDetector.fit(cdoc)

    BOWFlowConstructor.connect(cdoc)
    BOPFlowConstructor.connect(cdoc)
