from unittest import mock, TestCase
from unittest.mock import patch
# ------------------------------------
import sys
import os
sys.path.insert(0, os.path.dirname(sys.path[0]))
# ------------------------------------
from app import app
# from chunking.test_text import TestText
from ai import ChunkComparisonWithOriginalText

class Test_Sentence(TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    def setUp(self):
        self.url = '/chunking';

    def run_individual_test(self, test_text):
        with app.test_client() as client:
            result = client.post(self.url, json={"text_block": test_text})
            self.assertTrue(result.status_code == 200)
            chunks = result.json['data']
            error, data = ChunkComparisonWithOriginalText.run(test_text, chunks)
            print(data.similarity_score)
            print(data.difference_text)
            for detail in data.difference_details:
                print(detail)

    def test_chuking_paragragh_1(self):
        TestText = """
        Babylon was an ancient city located on the lower Euphrates river in southern Mesopotamia, within modern-day Hillah, Iraq, about 85 kilometers (55 miles) south of modern day Baghdad. Babylon functioned as the main cultural and political centre of the Akkadian-speaking region of Babylonia. Its rulers established two important empires in antiquity, the 19th–16th century BC Old Babylonian Empire, and the 7th–6th century BC Neo-Babylonian Empire. Babylon was also used as a regional capital of other empires, such as the Achaemenid Empire. Babylon was one of the most important urban centres of the ancient Near East, until its decline during the Hellenistic period. Nearby ancient sites are Kish, Borsippa, Dilbat, and Kutha.[2]

        The earliest known mention of Babylon as a small town appears on a clay tablet from the reign of Shar-Kali-Sharri (2217–2193 BC), of the Akkadian Empire.[3] Babylon was merely a religious and cultural centre at this point and neither an independent state nor a large city, subject to the Akkadian Empire. After the collapse of the Akkadian Empire, the south Mesopotamian region was dominated by the Gutian Dynasty for a few decades, before the rise of the Third Dynasty of Ur, which encompassed the whole of Mesopotamia, including the town of Babylon.

        The town became part of a small independent city-state with the rise of the first Babylonian Empire, now known as the Old Babylonian Empire, in the 19th century BC. The Amorite king Hammurabi founded the short-lived Old Babylonian Empire in the 18th century BC. He built Babylon into a major city and declared himself its king. Southern Mesopotamia became known as Babylonia, and Babylon eclipsed Nippur as the region's holy city. The empire waned under Hammurabi's son Samsu-iluna, and Babylon spent long periods under Assyrian, Kassite and Elamite domination. After the Assyrians destroyed and then rebuilt it, Babylon became the capital of the short-lived Neo-Babylonian Empire, from 626 to 539 BC. The Hanging Gardens of Babylon were ranked as one of the Seven Wonders of the Ancient World, allegedly existing between approximately 600 BC and AD 1. However, there are questions about whether the Hanging Gardens of Babylon even existed, as there is no mention within any extant Babylonian texts of its existence.
        
        After the fall of the Neo-Babylonian Empire, the city came under the rule of the Achaemenid, Seleucid, Parthian, Roman, Sassanid, and Muslim empires. The last known habitation of the town dates from the 11th century, when it was referred to as the "small village of Babel". It has been estimated that Babylon was the largest city in the world c. 1770 – c. 1670 BC, and again c. 612 – c. 320 BC. It was perhaps the first city to reach a population above 200,000.[6] Estimates for the maximum extent of its area range from 890 (3½ sq. mi.)[7] to 900 ha (2,200 acres).[8] The main sources of information about Babylon—excavation of the site itself, references in cuneiform texts found elsewhere in Mesopotamia, references in the Bible, descriptions in other classical writing, especially by Herodotus, and second-hand descriptions, citing the work of Ctesias and Berossus—present an incomplete and sometimes contradictory picture of the ancient city, even at its peak in the sixth century BC.[9] UNESCO inscribed Babylon as a World Heritage Site in 2019. The site receives thousands of visitors each year, almost all of whom are Iraqis.[10][11] Construction is rapidly increasing, which has caused encroachments upon the ruins.

        """
        self.run_individual_test(TestText)


    def test_chuking_single_long_sentence(self):
        TestText = """
        Babylon was an ancient city located on the lower Euphrates river in southern Mesopotamia, within modern-day Hillah, Iraq, about 85 kilometers (55 miles) south of modern day Baghdad. Babylon functioned as the main cultural and political centre of the Akkadian-speaking region of Babylonia. Its rulers established two important empires in antiquity, the 19th–16th century BC Old Babylonian Empire, and the 7th–6th century BC Neo-Babylonian Empire. Babylon was also used as a regional capital of other empires, such as the Achaemenid Empire. Babylon was one of the most important urban centres of the ancient Near East, until its decline during the Hellenistic period. Nearby ancient sites are Kish, Borsippa, Dilbat, and Kutha.[2] The earliest known mention of Babylon as a small town appears on a clay tablet from the reign of Shar-Kali-Sharri (2217–2193 BC), of the Akkadian Empire.[3] Babylon was merely a religious and cultural centre at this point and neither an independent state nor a large city, subject to the Akkadian Empire. After the collapse of the Akkadian Empire, the south Mesopotamian region was dominated by the Gutian Dynasty for a few decades, before the rise of the Third Dynasty of Ur, which encompassed the whole of Mesopotamia, including the town of Babylon. The town became part of a small independent city-state with the rise of the first Babylonian Empire, now known as the Old Babylonian Empire, in the 19th century BC. The Amorite king Hammurabi founded the short-lived Old Babylonian Empire in the 18th century BC. He built Babylon into a major city and declared himself its king. Southern Mesopotamia became known as Babylonia, and Babylon eclipsed Nippur as the region's holy city. The empire waned under Hammurabi's son Samsu-iluna, and Babylon spent long periods under Assyrian, Kassite and Elamite domination. After the Assyrians destroyed and then rebuilt it, Babylon became the capital of the short-lived Neo-Babylonian Empire, from 626 to 539 BC. The Hanging Gardens of Babylon were ranked as one of the Seven Wonders of the Ancient World, allegedly existing between approximately 600 BC and AD 1. However, there are questions about whether the Hanging Gardens of Babylon even existed, as there is no mention within any extant Babylonian texts of its existence. After the fall of the Neo-Babylonian Empire, the city came under the rule of the Achaemenid, Seleucid, Parthian, Roman, Sassanid, and Muslim empires. The last known habitation of the town dates from the 11th century, when it was referred to as the "small village of Babel". It has been estimated that Babylon was the largest city in the world c. 1770 – c. 1670 BC, and again c. 612 – c. 320 BC. It was perhaps the first city to reach a population above 200,000.[6] Estimates for the maximum extent of its area range from 890 (3½ sq. mi.)[7] to 900 ha (2,200 acres).[8] The main sources of information about Babylon—excavation of the site itself, references in cuneiform texts found elsewhere in Mesopotamia, references in the Bible, descriptions in other classical writing, especially by Herodotus, and second-hand descriptions, citing the work of Ctesias and Berossus—present an incomplete and sometimes contradictory picture of the ancient city, even at its peak in the sixth century BC.[9] UNESCO inscribed Babylon as a World Heritage Site in 2019. The site receives thousands of visitors each year, almost all of whom are Iraqis.[10][11] Construction is rapidly increasing, which has caused encroachments upon the ruins.
        """
        self.run_individual_test(TestText)


    def test_chunking_small_sentences(self):
        TestText = """
        Babylon was an ancient city located on the lower Euphrates river in southern Mesopotamia, within modern-day Hillah, Iraq, about 85 kilometers (55 miles) south of modern day Baghdad.
        Babylon functioned as the main cultural and political centre of the Akkadian-speaking region of Babylonia.
        Its rulers established two important empires in antiquity, the 19th–16th century BC Old Babylonian Empire, and the 7th–6th century BC Neo-Babylonian Empire. 
        Babylon was also used as a regional capital of other empires, such as the Achaemenid Empire.
        Babylon was one of the most important urban centres of the ancient Near East, until its decline during the Hellenistic period. Nearby ancient sites are Kish, Borsippa, Dilbat, and Kutha.
        The earliest known mention of Babylon as a small town appears on a clay tablet from the reign of Shar-Kali-Sharri (2217–2193 BC), of the Akkadian Empire.
        Babylon was merely a religious and cultural centre at this point and neither an independent state nor a large city, subject to the Akkadian Empire.
        After the collapse of the Akkadian Empire, the south Mesopotamian region was dominated by the Gutian Dynasty for a few decades, before the rise of the Third Dynasty of Ur, which encompassed the whole of Mesopotamia, including the town of Babylon.

        
        The town became part of a small independent city-state with the rise of the first Babylonian Empire, now known as the Old Babylonian Empire, in the 19th century BC. 
        The Amorite king Hammurabi founded the short-lived Old Babylonian Empire in the 18th century BC. 
        He built Babylon into a major city and declared himself its king. 
        Southern Mesopotamia became known as Babylonia, and Babylon eclipsed Nippur as the region's holy city. 
        The empire waned under Hammurabi's son Samsu-iluna, and Babylon spent long periods under Assyrian, Kassite and Elamite domination. 
        After the Assyrians destroyed and then rebuilt it, Babylon became the capital of the short-lived Neo-Babylonian Empire, from 626 to 539 BC. 
        The Hanging Gardens of Babylon were ranked as one of the Seven Wonders of the Ancient World, allegedly existing between approximately 600 BC and AD 1. 
        However, there are questions about whether the Hanging Gardens of Babylon even existed, as there is no mention within any extant Babylonian texts of its existence. 
        After the fall of the Neo-Babylonian Empire, the city came under the rule of the Achaemenid, Seleucid, Parthian, Roman, Sassanid, and Muslim empires. 
        The last known habitation of the town dates from the 11th century, when it was referred to as the "small village of Babel". 
        It has been estimated that Babylon was the largest city in the world c. 1770 – c. 1670 BC, and again c. 612 – c. 320 BC. 
        It was perhaps the first city to reach a population above 200,000. 
        Estimates for the maximum extent of its area range from 890 (3½ sq. mi.)[7] to 900 ha (2,200 acres). 
        The main sources of information about Babylon—excavation of the site itself, references in cuneiform texts found elsewhere in Mesopotamia, references in the Bible, descriptions in other classical writing, especially by Herodotus, and second-hand descriptions, citing the work of Ctesias and Berossus—present an incomplete and sometimes contradictory picture of the ancient city, even at its peak in the sixth century BC.
        UNESCO inscribed Babylon as a World Heritage Site in 2019. 
        The site receives thousands of visitors each year, almost all of whom are Iraqis. 
        Construction is rapidly increasing, which has caused encroachments upon the ruins.
        """
        self.run_individual_test(TestText)
