  
When I use the sequences from DeepMSA2 to obtain a richer and more diverse sequence space, I sometimes try to combine all the qMSA, dMSA, and related outputs from the raw DeepMSA results. The idea is to merge them into a single compiled set of sequences. However, this may not always be ideal, since some headers are the same but contain multiple sequence contents—likely due to differences in the MSA alignment algorithms. This can introduce noise into the sequence purification process. To avoid that, I would select the MSA file from the DeepMSA2 output that has the largest number of sequences and use that as the input. 

such as  Duplicate header found: renaming '>UniRef100_A0A353S8P3' to '>UniRef100_A0A353S8P3_2'
  Duplicate header found: renaming '>UniRef100_A0A3A6HTF8' to '>UniRef100_A0A3A6HTF8_2'
  Duplicate header found: renaming '>UniRef100_A0A3E2T1Y8' to '>UniRef100_A0A3E2T1Y8_2'
  Duplicate header found: renaming '>UniRef100_A0A6H1WWP5' to '>UniRef100_A0A6H1WWP5_2'
  Duplicate header found: renaming '>UniRef100_A0A6N7VUW0' to '>UniRef100_A0A6N7VUW0_2'
  Duplicate header found: renaming '>UniRef100_A0A7C6FU87' to '>UniRef100_A0A7C6FU87_2'
  Duplicate header found: renaming '>UniRef100_A0A7X2NR29' to '>UniRef100_A0A7X2NR29_2'
  Duplicate header found: renaming '>UniRef100_A0A8J7KSL9' to '>UniRef100_A0A8J7KSL9_2'
  Duplicate header found: renaming '>UniRef100_UPI00042A3319' to '>UniRef100_UPI00042A3319_2'
  Duplicate header found: renaming '>UniRef100_UPI000829F604' to '>UniRef100_UPI000829F604_2'
  Duplicate header found: renaming '>UniRef100_UPI00156D8A9E' to '>UniRef100_UPI00156D8A9E_2'
  Duplicate header found: renaming '>UniRef100_UPI001A9C1000' to '>UniRef100_UPI001A9C1000_2'
  Duplicate header found: renaming '>UniRef100_A0A847Z7W2' to '>UniRef100_A0A847Z7W2_2'
  Duplicate header found: renaming '>UniRef100_A0A0R2R946' to '>UniRef100_A0A0R2R946_2'
  Duplicate header found: renaming '>UniRef100_A0A0R2SK18' to '>UniRef100_A0A0R2SK18_2'
  Duplicate header found: renaming '>UniRef100_A0A0R2SR59' to '>UniRef100_A0A0R2SR59_2'

for example
>UniRef100_UPI001A9C1000_2
-----------LDKESVKNILLVGLPLSLQDAFVNISFLIITSVINTIGVIASASVGVVGKIIMFAMLPPISFGSAVSVMTAQNIGAGEHKRARKVLYYGILFSLIFGIFATLYSQFYPETLTSIFSNDIEVINSSNQYLMSFSIDCIMVSFVFCMNGYLSGIGKSIVSLIHSLIATGVRIPLTYILNKTAGVTLYELGLAAPISTFVSILICFIYLYWTYRKDKLNYNDNNINIECDENTILL-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
>UniRef100_UPI001A9C1000
-------MSNLTRGKISTTLLKFAIPFLFASLLQALYGAVDLFVVGRFNSATVSAVSIGSQVMQTVTGIILGISMGGTVLIGKRIGEKNDDGAAKAIGSLSILFIIFTIILTPLMLLFTNSAISLMHTPLEAVNYTKQYIIICSLGIPFIIGYNSISGIFRGLGDSKTPVYFIAIACVINIIVDFILIGIFNFGAVGAAIATTSSQAISFL-IAVIYMIKKGFSFEINKKHFKLDKESVKNILLVGLPLSLQDAFVNISFLIITSVINTIG-VIASASVGVVGKIIMFAMLPPISFGSAVSVMTAQNIGAGEHKRARKVLYYGILFSLIFGIFATLYSQFYPETLTSIFS--NDIEVINSS-NQYLMSFSIDCIMVSFVFCMNGYLSGIGKSIVSLIHSLIATFGVRIPLTYILNKTAGVTLYELGLAAPISTFVSILICFIYL-----------------

It’s clear that one of the alignments is incorrect, since it only matches part of the query sequence, even though they are the same protein with the same structural domain (and not a multi-domain protein). After checking the DeepMSA2 paper, I think this might be a limitation of the method on either dMSA or qMSA. Therefore, having more sequences in the file does not necessarily mean better results. Filtering and careful MSA selection are still important, as I want to avoid cases where the same sequence appears twice but is aligned differently.