import numpy as np
from os.path import join, dirname
HERE = dirname(__file__)

class CCode:

    def __init__(self, code, header):
        self.code = code
        self.header = header


class CArray(CCode):
    def __init__(self, name, a, const=False, declared=True, define_at_declare=False, do_initialize=None):
        if not isinstance(a, np.ndarray):
            assert isinstance(a, tuple)
            a = np.zeros(a)
            do_initialize = False if do_initialize is None else do_initialize
        else:
            do_initialize = True if do_initialize is None else do_initialize
        self.shape = tuple([int(s) for s in a.shape])
        assert len(a.shape) == 2, 'We assume 2D arrays.'

        shape_name = name + '_shape'
        shape_declaration = 'int {name}[{size}]'.format(
            name=shape_name,
            size=len(a.shape),
        )

        if define_at_declare:
            shape_definition = '{{{data}}}'.format(
                name=shape_name,
                size=len(a.shape),
                data=', '.join(['%d' % s for s in a.shape]),
            )
            shape_declaration = '{decl} = {defi}'.format(
                decl=shape_declaration,
                defi=shape_definition
            )
            shape = ''
        else:
            shape = '\n'.join([
                '{name}_shape[{i}] = {v};'.format(
                    name=name,
                    i=i,
                    v=v,
                )
                for i, v in enumerate(a.shape)
            ])

        data_name = name
        dtypename = 'int' if a.dtype == 'int' else 'double'
        data_declaration = '{const}{dtypename} {name}[{size}]'.format(
            const='const ' if const else '',
            dtypename=dtypename,
            size=a.size,
            name=data_name,
        )
        if define_at_declare:
            data_def = '{{{data}}}'.format(
                name=data_name,
                dtypename=dtypename,
                size=a.size,
                data=', '.join(['%.60g' % f for f in a.ravel()]),
            )
            data_declaration = '{decl} = {defi}'.format(
                decl=data_declaration,
                defi=data_def,
            )
            data = ''
        else:
            if not do_initialize:
                data = ''
            else:
                data = '\n'.join([
                    '{name}[{i}] = {v};'.format(
                        name=name,
                        i=i,
                        v='%.60g' % v
                    )
                    for i, v in enumerate(a.ravel())
                ])

        self.code = shape + ';\n' + data + ';\n'
        self.actual_header = shape_declaration + ';\n' + data_declaration + ';\n'
        self.header = self.actual_header if declared else ''
        self.name = name


class Layer(CCode):

    def __init__(self, i, activation):
        actstr = str(activation).lower()
        relu = 'relu' in actstr
        tanh = 'tanh' in actstr
        linear = (not relu) and (not tanh)
        self.code = '''
        mmult(
            preactivation_{i}, preactivation_{i}_shape[0], preactivation_{i}_shape[1],
            kernel_{i}, kernel_{i}_shape[0], kernel_{i}_shape[1],
            xA_{i}
            );
        bias_add(
            xA_{i}, xA_{i}_shape[0], xA_{i}_shape[1],
            bias_{i},
            preactivation_{iPlusOne}
        );
        activate(
            preactivation_{iPlusOne}, 
            preactivation_{iPlusOne}_shape[0],
            preactivation_{iPlusOne}_shape[1],
            {act_type}
        );
        '''.format(
            i=i, iPlusOne=i+1,
            act_type=(0 if relu else (1 if tanh else 2)),
        )
        self.header = ''
        self.name = 'layer_%d' % i
        self.i = i


def get_utils_code(header_guards=False):
    with open(join(HERE, 'mm_utils.h'), 'r') as fp:
        header = fp.read()

    with open(join(HERE, 'mm_utils.c'), 'r') as fp:
        code = fp.read()

    if not header_guards:
        # Remove header_guards.
        header = '\n'.join([
            line
            for line in header.split('\n')
            if ('#ifndef' not in line and '#endif' not in line)
        ])
    
    return CCode(code, header)


FORMAT_CODE_MLP_FUNC = '''
void MLP(
    const double *preactivation_0,
    double *preactivation_{nlayers}
    ) {{
    const int preactivation_0_shape[2] = {{{batch_size}, {n_inputs}}};
    const int preactivation_{nlayers}_shape[2] = {{{batch_size}, {n_outputs}}};
    {statements}
}}
'''

FORMAT_HEADER_MLP_FUNC = '''
void MLP(
    const double *preactivation_0,
    double *preactivation_{nlayers}
);
'''

FORMAT_CODE_MAIN = '''#include "MLP.h"
{utils_header}

{globals}

{func}
'''

FORMAT_HEADER_MAIN = '''#ifndef MLP_H
#define MLP_H
{globals}

{func}
#endif //MLP_H
'''

class CModel(CCode):
    def __init__(self, model, batch_size, add_utils_code=True):

        arrays_to_write = []

        kernels = []
        biases = []
        xAs = []

        statements = []

        for i, layer in enumerate(model.layers):
            kernel, bias = [x.numpy() for x in layer.weights]

            arrays_to_write.append(CArray('kernel_%d' % i, kernel))
            kernels.append(arrays_to_write[-1])

            arrays_to_write.append(
                CArray('bias_%d' % i, bias.reshape((1, -1))))
            biases.append(arrays_to_write[-1])

            act_shape = (batch_size, kernel.shape[1])
            arrays_to_write.append(CArray('xA_%d' % i, act_shape, const=False))
            xAs.append(arrays_to_write[-1])

            is_last_layer = i == len(model.layers) - 1
            arrays_to_write.append(CArray('preactivation_%d' % (
                i+1,), act_shape, const=False, declared=not is_last_layer))

            statements.append(Layer(i, layer.activation))

        n_inputs = kernels[0].shape[0]
        n_outputs = kernels[-1].shape[1]

        main_function_code = CCode(FORMAT_CODE_MLP_FUNC.format(
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            batch_size=batch_size,
            nlayers=len(model.layers),
            statements='\n'.join([st.code for st in statements])
        ),
            FORMAT_HEADER_MLP_FUNC.format(
            nlayers=len(model.layers),
            statements='\n'.join([st.code for st in statements])
        )
        )

        if add_utils_code:
            utils = get_utils_code()

        setup_statements = '\n'.join([
            c.code for c in arrays_to_write if c.header != ''
        ])
        global_code = '''void setup() {
            %s
        }''' % setup_statements

        if add_utils_code:
            global_code = utils.code + '\n\n' + global_code 

        global_definitions_header = (
            '\n'.join([c.header for c in arrays_to_write])
            + 'void setup();'
        )

        if add_utils_code:
            global_definitions_header = utils.header + '\n\n' + global_definitions_header


        self.code = FORMAT_CODE_MAIN.format(
            utils_header='#include "mm_utils.h"' if not add_utils_code else '',
            globals=global_code, 
            func=main_function_code.code,
        )

        self.header = FORMAT_HEADER_MAIN.format(globals=global_definitions_header, func=main_function_code.header)

    def save(self, name='MLP'):
        header_name = name + '.h'
        code_name = name + '.c'
        with open(code_name, 'w') as fp:
            fp.write(self.code.replace('MLP', name))
        with open(header_name, 'w') as fp:
            fp.write(self.header.replace('MLP', name))
