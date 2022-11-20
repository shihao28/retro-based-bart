from transformers import TrainingArguments

class Config(object):
    
    train=dict(

        data=dict(
            path='eli5',
            # name='20220301.en'
        ),

        test_size=0.3,

        model=dict(
            name='facebook/bart-base',
            args=dict(
                add_cross_attention=False
            )
        ),

        training_args=dict(
            output_dir="./results",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            weight_decay=0.01,       
        ),

        epoch=2,
        batch_size=2,
        device='cuda',
    )
