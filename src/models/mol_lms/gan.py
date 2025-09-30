import torch
import torch.optim as optim
from generator import Generator
from discriminator import Discriminator
from copy import deepcopy
import numpy as np
import time
import transformers
import json
from token_splits import pretokenizer_dict

class Gan:
    """Class for training and evaluation of generator and discriminator models."""


    def __init__(self, model_directory, tokenizer_directory, tokenizer_type='bert', mutation_parameter=0.5, lr=0.00001, device="cpu", saved_generator=None, saved_discriminator=None, generator_only=False, top_k=5, random_init=False):
        """Constructor for Gan class.
        
        Args:
            model_directory (str): Directory to be used to initialize models using hugging face
            tokenizer (hugging face tokenizer): Tokenizer determines conversion of text to token ids 
            mutation_parameter (float): probability of a token being replaced by a mask for input to generator
            lr (float): learning rate for AdamW optimizer
            device (str): device for training
            saved_generator (pytorch model): weights to initialize generator
            saved_discriminator (pytorch model): weights to initialize discriminator
        """
        super().__init__()

        # initialize class data
        self._device = torch.device(device)
        self._lr = lr
        self._mutation_parameter = mutation_parameter
        self._top_k = top_k

        # initialize tokenizer
        self._tokenizer = None
        try:
            with open(tokenizer_directory + '/config.json', 'r') as f:
                tokenizer_config = json.load(f)
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_directory, **tokenizer_config)
        except:
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_directory, use_auth_token=True)
        self._tokenizer.backend_tokenizer.pre_tokenizer = pretokenizer_dict[tokenizer_type]
        
        # allows case with only generator for evaluation
        self._gen = Generator(model_directory, self._tokenizer, random_init).to(self._device)
        self._disc = None
        self._optimizer_disc = None
        self._optimizer_gen = None
        self._criterion = torch.nn.BCEWithLogitsLoss()
        self._optimizer_gen = optim.AdamW(self._gen.parameters(), lr=self._lr)
        if not generator_only:
            self._disc = Discriminator(model_directory, random_init).to(self._device)
            self._optimizer_disc = optim.AdamW(self._disc.parameters(), lr=self._lr)

        if saved_generator != None:
            self._gen.load_state_dict(torch.load(saved_generator))

        # note: this will fail if it conflicts with generator_only
        if saved_discriminator != None:
            self._disc.load_state_dict(torch.load(saved_discriminator))

    @property
    def generator_only(self):
        """Get value for generator only property.

        Returns:
            bool that determines whether gan is generator only

        """
        return (self._disc is None)

    def train_step(self, smiles_batch):
        """Perform training step based on a batch of smiles.
        
        Args:
            batch (List[str]): List of smiles strings for molecules

        Returns:
            Tuple with discriminator loss and generator loss
        """
        # tokenize smiles
        batch = self._tokenizer(smiles_batch, padding=True, return_tensors='pt')

        # set models to training mode
        self._gen.train()

        # used to fill input for loss calculation
        real_label = 1.
        fake_label = 0. 

        # track losses
        metric_disc_loss = 0.0
        metric_gen_loss = 0.0

        # update D
        if self._disc is not None:
            self._disc.train()
            self._disc.zero_grad()

            # determine loss on disc from real data
            batch_ids = batch['input_ids'].to(self._device)
            batch_mask = batch['attention_mask'].to(self._device)
            output = self._disc(input_ids=batch_ids, one_hot_tokens=None, attention_mask=batch_mask).view(-1)
            label = torch.full((batch_ids.shape[0],), real_label, dtype=torch.float, device=self._device)
            err_disc_real = self._criterion(output, label)
            err_disc_real.backward()

            # determine loss on disc from fake data
            # task = np.random.choice(['replace','insert','delete','combine'])
            batch_ids, batch_mask, _ = self.generate_masks(batch['input_ids'], batch['attention_mask'], 'replace')

            batch_ids = batch_ids.to(self._device)
            batch_mask = batch_mask.to(self._device)
            fake = self._gen(input_ids=batch_ids, attention_mask=batch_mask)
            label.fill_(fake_label)
            output = self._disc(input_ids=None, one_hot_tokens=fake.detach(), attention_mask=batch_mask).view(-1)
            err_disc_fake = self._criterion(output, label)
            err_disc_fake.backward()

            # update disc
            self._optimizer_disc.step()
            metric_disc_loss = (1.0 + err_disc_fake.item() + err_disc_real.item()) / 2.0

            # determine loss on generator
            self._gen.zero_grad()
            label.fill_(real_label)
            output = self._disc(input_ids=None, one_hot_tokens=fake, attention_mask=batch_mask).view(-1)
            err_gen = self._criterion(output, label)
            err_gen.backward()
            self._optimizer_gen.step()       

            metric_gen_loss = 1.0 + err_gen.item()

        else:
            self._gen.zero_grad()
            mlm_loss_fct = torch.nn.CrossEntropyLoss()  # -100 index = padding token
            batch_ids, batch_mask, batch_labels = self.generate_masks(batch['input_ids'], batch['attention_mask'], 'replace')
            batch_ids = batch_ids.to(self._device)
            batch_mask = batch_mask.to(self._device)
            batch_labels = batch_labels.to(self._device)
            fake = self._gen(input_ids=batch_ids, attention_mask=batch_mask, hard=False, raw=True)
            masked_lm_loss = mlm_loss_fct(fake.view(-1, self._gen.embedding.config.vocab_size), batch_labels.view(-1))
            masked_lm_loss.backward()
            self._optimizer_gen.step()  
            metric_gen_loss = masked_lm_loss.item()

        return (metric_disc_loss, metric_gen_loss)

    def train_epoch(self, dataloader):
        """Perform training step based on a batch of smiles.
        
        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader to iterate through dataset

        Returns:
            Tuple with time, discriminator loss, and generator loss
        """
        # initial values
        t = time.time()
        metric_disc_loss = 0.0
        metric_gen_loss = 0.0 
        batch_counter = 0

        # iterate through dataloader and train
        for _,batch in enumerate(dataloader):
            (batch_disc_loss, batch_gen_loss) = self.train_step(batch)
            metric_disc_loss += batch_disc_loss
            metric_gen_loss += batch_gen_loss
            batch_counter += 1

        return (time.time() - t), metric_disc_loss / batch_counter, metric_gen_loss / batch_counter

    def generate_masks(self, batch_ids, batch_mask, task='replace'):
        """Randomly mask token ids for use in generation

        Args:
            batch_ids (tensor): token ids for molecules sequences
            batch_mask (tensor): attention mask for molecule sequences
            task (str): replace, insert, or delete

        Returns:
            tensor with randomly masksed token ids
            tensor with updated attention mask
        """

        # make copies of original data
        masked_ids = deepcopy(batch_ids)
        updated_attention_mask = deepcopy(batch_mask)
        masked_ids_cols = len(masked_ids[0])

        # padding to allow larger molecules through recombination or insertion
        if task == "combine":
            if masked_ids_cols < self._tokenizer.model_max_length:
                updated_masked_ids_cols = min(self._tokenizer.model_max_length, 2*masked_ids_cols-1)
                masked_ids = torch.nn.functional.pad(masked_ids, (0,updated_masked_ids_cols-masked_ids_cols), 'constant', 0)
                updated_attention_mask = torch.nn.functional.pad(updated_attention_mask, (0,updated_masked_ids_cols-masked_ids_cols), 'constant', 0)
                masked_ids_cols = updated_masked_ids_cols
        elif task == "insert":
            if masked_ids_cols < self._tokenizer.model_max_length:
                masked_ids = torch.nn.functional.pad(masked_ids, (0,1), 'constant', 0)
                updated_attention_mask = torch.nn.functional.pad(updated_attention_mask, (0,1), 'constant', 0)
                masked_ids_cols = masked_ids_cols + 1

        # sets for insert/delete tasks
        insert_set = set()
        delete_set = set()

        for i in range(len(masked_ids)):

            # for combine task, sample another molecules
            if task == "combine":

                # determine lenghts for parents
                parent_length = torch.count_nonzero(masked_ids[i]).item()
                second_parent_index = np.random.choice(len(masked_ids))
                second_parent_length = torch.count_nonzero(masked_ids[second_parent_index]).item()

                # check if either parent is empty
                if (parent_length > 2) and (second_parent_length > 2) and (masked_ids_cols > 4):
                    end_index = np.random.choice(np.arange(2,parent_length))
                    end_index = min(masked_ids_cols-3, end_index)
            
                    start_index = np.random.choice(np.arange(1,second_parent_length-1))
                    start_index = max(end_index+1+second_parent_length-masked_ids_cols, start_index)
                    updated_length = second_parent_length - start_index

                    # overwrite masked_ids with combination
                    temp_ids = torch.zeros_like(masked_ids[i])
                    temp_ids[:end_index] = masked_ids[i,:end_index]
                    temp_ids[end_index] = self._tokenizer.mask_token_id
                    temp_ids[end_index+1:end_index+updated_length+1] = masked_ids[second_parent_index,start_index:second_parent_length]
                    masked_ids[i] = temp_ids

                    updated_attention_mask[i,:] = 0
                    updated_attention_mask[i,:end_index+updated_length+1] = 1

            # each sequence has CLS and SEP tokens at beginning and end
            number_of_tokens = torch.count_nonzero(masked_ids[i]) - 2

            if number_of_tokens == 0:
                # corner case for empty inputs
                if len(masked_ids[i]) > 2:
                    masked_ids[i][0] = self._tokenizer.cls_token_id
                    masked_ids[i][1] = self._tokenizer.mask_token_id
                    masked_ids[i][2] = self._tokenizer.sep_token_id
                    updated_attention_mask[i][:3] = 1
            else:
                # binomial distribution based off mutation_parameter with a minumum of 1 mask
                number_of_mutations = np.random.binomial(number_of_tokens, self._mutation_parameter)
                number_of_mutations = max(1, number_of_mutations)
                mutation_locations = set(np.random.choice(np.arange(1,number_of_tokens+1), number_of_mutations, replace=False))

                if task == 'insert':
                    if (number_of_tokens + 2) < len(masked_ids[i]):
                        insert_set.add(mutation_locations.pop())
                elif task == 'delete':
                    if number_of_tokens > 1:
                        selected_delete_location = mutation_locations.pop()
                        if selected_delete_location < number_of_tokens:
                            delete_set.add(selected_delete_location + 1)
                        mutation_locations.add(selected_delete_location)

                # apply mutations
                for location in mutation_locations:
                    masked_ids[i][location] = self._tokenizer.mask_token_id

                # apply insertion and deletion if specified
                if len(insert_set) > 0:
                    insert_location = insert_set.pop()
                    updated_attention_mask[i, number_of_tokens+2] = 1
                    temp_ids = torch.zeros_like(masked_ids[i])
                    temp_ids[:insert_location] = masked_ids[i,:insert_location]
                    temp_ids[insert_location] = self._tokenizer.mask_token_id
                    temp_ids[insert_location+1:] = masked_ids[i,insert_location:-1]
                    masked_ids[i] = temp_ids
                elif len(delete_set) > 0:
                    delete_location = delete_set.pop()
                    updated_attention_mask[i, number_of_tokens+1] = 0
                    temp_ids = torch.zeros_like(masked_ids[i])
                    temp_ids[:delete_location] = masked_ids[i,:delete_location]
                    temp_ids[delete_location:-1] = masked_ids[i,delete_location+1:]
                    masked_ids[i] = temp_ids          

        # labels based on masks
        labels = None
        if task == 'replace':
            labels = torch.where(masked_ids == self._tokenizer.mask_token_id, batch_ids, -100)

        return masked_ids, updated_attention_mask, labels

    def evaluate_generator(self, smiles_batch):
        """Generate text sequences from a batch of smiles.
        
        Args:
            batch (List[str]): List of smiles strings for molecules

        Returns:
            List[str] with generated molecules
        """
        self._gen.eval()
        masked_sequences = []
        with torch.no_grad():
            # tokenize batch
            batch = self._tokenizer(smiles_batch, padding=True, return_tensors='pt')

            # generate random masks
            # task = np.random.choice(['replace','insert','delete','combine'])
            task = np.random.choice(['replace','insert','delete'])
            batch_ids, batch_mask, _ = self.generate_masks(batch['input_ids'], batch['attention_mask'], task)
            batch_ids = batch_ids.to(self._device)
            batch_mask = batch_mask.to(self._device)
            
            # generate token probabilities for masked inputs
            fake = self._gen(input_ids=batch_ids, attention_mask=batch_mask, hard=False).detach().cpu()
            batch_ids = batch_ids.detach().cpu()

            results = []
            for i in range(fake.size(0)):

                # masked sequence
                input_ids = batch_ids[i]

                masked_sequences.append(self._tokenizer.decode(input_ids))

                # find probablities at locations with a mask token
                masked_index = torch.nonzero(input_ids == self._tokenizer.mask_token_id, as_tuple=False).flatten()
                probs = fake[i, masked_index, :]

                # find topk predictions for each mask token
                values, predictions = probs.topk(self._top_k)

                possible_indices = torch.zeros(len(predictions), dtype=torch.long)
                for k in range(self._top_k):
                    indices = None
                    if k == 0:
                        # take top predictions
                        indices = predictions[:,0]
                    else:
                        # find next best prediction
                        max_score = -1
                        best_index = -1
                        for j in range(len(predictions)):
                            current_indices = possible_indices.detach().clone()
                            current_indices[j] += 1
                            current_score = torch.prod(torch.gather(values, 1, current_indices.unsqueeze(1)))
                            if current_score > max_score:
                                max_score = current_score
                                best_index = j

                        if best_index == -1:
                            break

                        possible_indices[best_index] += 1
                        indices = torch.gather(predictions, 1, possible_indices.unsqueeze(1)).flatten()

                    # fill in masks with predictions
                    input_ids[masked_index] = indices
                    results.append(self._tokenizer.decode(input_ids, skip_special_tokens=True).replace(' ','').replace('##',''))

            return results, masked_sequences
    
    def save(self, generator_file, discriminator_file):
        """Save generator and discriminator."""
        if self._disc is None:
            raise AttributeError('Discriminator does not exist')
        torch.save(self._gen.state_dict(), generator_file)
        torch.save(self._disc.state_dict(), discriminator_file)







