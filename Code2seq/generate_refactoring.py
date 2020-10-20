import os,random
from shutil import copyfile
from refactoring_methods import *


def generate_adversarial(k, code, file_name):
        final_refactor = ''
        function_list = []
        class_name = ''
        vv = 0

        Class_list, raw_code =  extract_class(code)

        for class_name in Class_list:
            function_list, class_name = extract_function(class_name)

        refac = []
        for code in function_list:

            new_rf = code
            new_refactored_code = code

            for t in range(k):
                print(t)
                # refactors_list = [  rename_argument,
                #                     add_argumemts,
                #                     rename_api,
                #                     rename_local_variable,
                #                     add_local_variable,
                #                     rename_method_name
                #                     ]
                refactors_list = [rename_argument, return_optimal, add_argumemts,rename_api,
                                    rename_local_variable,add_local_variable,rename_method_name, enhance_for_loop,enhance_filed,enhance_if,add_print]#
                vv = 0
                # if code == new_refactored_code:
                #     print('im in if')

                while new_rf == new_refactored_code and vv <= 20:
                    try:
                        vv += 1
                        refactor       = random.choice(refactors_list)
                        print('*'*50 , refactor , '*'*50)
                        new_refactored_code = refactor(new_refactored_code)
                        # print(vv)

                    except Exception as error:
                        print('error:\t',error)

                new_rf = new_refactored_code

                print('----------------------------OUT of WHILE----------------------------------', vv)
                print('----------------------------CHANGED THJIS TIME:----------------------------------', vv)
            refac.append(new_refactored_code)
        code_body = raw_code.strip() + ' ' + class_name.strip()

        for i in range(len(refac)):
            final_refactor = code_body.replace('vesal'+ str(i), str(refac[i]))
            code_body = final_refactor


        return final_refactor


if __name__ == '__main__':
    try:
        K = 1
        open_file = open('Matcher.java','r', encoding = 'ISO-8859-1')
        code = open_file.read()
        filename = 'matcher.java'

        new_code = generate_adversarial(K, code, filename)

        wr_path = 'new_' + filename

        if new_code is not '':
            l = open(wr_path,'w')
            l.write(new_code)
        # i= 0
    #
    #     mode = '' # Options: training, test
    #     # new = 'new/'
    #     # dest_foler = '/Users/Vesal/Desktop/java-small/test/'
    #     dest_foler = '/Users/Vesal/Desktop/g/'
    #     # source = '/Users/Vesal/Desktop/vvv/'
    #     source = '/Users/Vesal/Desktop/g/'
    #     o = count = 0
    #
    #     # open_file = open(source,'r')
    #     # code = open_file.read()
    #     # generate_adversarial(K, source, code, 'ProductSchemaDemo.java')
    #     print(source + mode)
    #     for path, d, file_names in os.walk(source + mode):
    #         # if filename != '/Users/Vesal/Desktop/java-large/training/aphyr__clj-antlr/demo/checkouts/clj-antlr/demo/checkouts/clj-antlr/demo/checkouts/clj-antlr/demo/checkouts/clj-antlr/demo/checkouts/clj-antlr/demo/checkouts/clj-antlr/demo/checkouts/clj-antlr/demo/checkouts/clj-antlr/demo/checkouts/clj-antlr/demo/checkouts/clj-antlr/demo/checkouts/clj-antlr/demo/checkouts/clj-antlr/demo/checkouts/clj-antlr/demo/checkouts/clj-antlr/demo/checkouts/clj-antlr/demo/checkouts/clj-antlr/demo/checkouts/clj-antlr/demo/checkouts/clj-antlr/demo/checkouts/clj-antlr/demo/checkouts/clj-antlr/demo/checkouts/clj-antlr/demo/checkouts/clj-antlr/demo/checkouts/clj-antlr/demo/checkouts/clj-antlr/demo/checkouts/clj-antlr/demo/checkouts/clj-antlr/demo/checkouts/clj-antlr/demo/checkouts/clj-antlr/demo/checkouts/clj-antlr/demo/checkouts/clj-antlr/demo/checkouts/clj-antlr/demo/checkouts/clj-antlr/demo/checkouts/clj-antlr/demo/checkouts/clj-antlr/demo/checkouts/clj-antlr/demo/checkouts/clj-antlr/demo/checkouts/clj-antlr/demo/checkouts/clj-antlr/demo/src/java/EdnListener.java':
    #             for filename in file_names:
    #                 # print(filename)
    #                 if '.java' in filename:
    #                     print('ves')
    #                     # if 'new_' in filename:
    #                         # new = 'new_'
    #                     #     correct_name = "".join(filename.rsplit(new))
    #                     #     copyfile(path +'/'+ filename, source + mode + dest_foler + '/' + correct_name)
    #
    #                     # print(filename)
    #                     # copyfile(path +'/'+ filename, dest_foler + mode + filename)
    #                     o += 1
    #                     # i += 1
    #                     # print(i)
    #                     # os.rename(path +'/'+ filename, source + mode + '/' + filename)
    #                     try:
    #                         open_file = open(path +'/'+ filename,'r', encoding = 'ISO-8859-1')
    #                         code = open_file.read()
    #                         new_code = generate_adversarial(K, code, filename)
    #
    #                         # wr_path = dest_foler + mode + 'new/' + filename
    #                         wr_path = dest_foler + mode + 'new_' + filename
    #
    #                         if new_code is not '':
    #                             count += 1
    #                             l = open(wr_path,'w')
    #                             l.write(new_code)
    #
    #                         else:
    #                             print('IN ERRRRORORRORORORORO')
    #                             print(filename)
    #                             print(new_code)
    #                             l = open(wr_path,'w')
    #                             l.write(code)
    #
    #                     except Exception as error:
    #                         l = open(wr_path,'w')
    #                         l.write(code)
    #
    #     print('done with for')
    #     print('Total files:', o)
    #     print('Total refactored files:', count)
    #
    except OSError as exc:
        print('ERRORR----Till NOW:')
    #     print('Total copied files:', o)
    #     print('Total refactored files:', count)
