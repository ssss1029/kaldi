// gmmbin/gmm-init-mono.cc

// Copyright 2009-2011  Microsoft Corporation

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.


#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/hmm-topology.h"
#include "hmm/transition-model.h"

namespace kaldi {
// This function reads a file like:
// 1 2 3
// 4 5
// 6 7 8
// where each line is a list of integer id's of phones (that should have their pdfs shared).
void ReadSharedPhonesList(std::string rxfilename, std::vector<std::vector<int32> > *list_out) {
  list_out->clear();
  Input input(rxfilename);
  std::istream &is = input.Stream();
  std::string line;
  while (std::getline(is, line)) {
    list_out->push_back(std::vector<int32>());
    if (!SplitStringToIntegers(line, " \t\r", true, &(list_out->back())))
      KALDI_ERR << "Bad line in shared phones list: " << line << " (reading "
                << PrintableRxfilename(rxfilename) << ")";
    std::sort(list_out->rbegin()->begin(), list_out->rbegin()->end());
    if (!IsSortedAndUniq(*(list_out->rbegin())))
      KALDI_ERR << "Bad line in shared phones list (repeated phone): " << line
                << " (reading " << PrintableRxfilename(rxfilename) << ")";
  }
}

} // end namespace kaldi

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using kaldi::int32;

    const char *usage =
        "Initialize monophone GMM.\n"
        "Usage:  gmm-init-mono <topology-in> <dim> <model-out> <tree-out> \n"
        "e.g.: \n"
        " gmm-init-mono topo 39 mono.mdl mono.tree\n";

    bool binary = true;
    std::string train_feats;
    std::string shared_phones_rxfilename;
    BaseFloat perturb_factor = 0.0;
    ParseOptions po(usage);
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("train-feats", &train_feats,
                "rspecifier for training features [used to set mean and variance]");
    po.Register("shared-phones", &shared_phones_rxfilename,
                "rxfilename containing, on each line, a list of phones whose pdfs should be shared.");
    po.Register("perturb-factor", &perturb_factor,
                "Perturb the means using this fraction of standard deviation.");
    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }


    std::string topo_filename = po.GetArg(1);
    int dim = atoi(po.GetArg(2).c_str());      // Convert to integer. Usually 39.
    KALDI_ASSERT(dim> 0 && dim < 10000);
    std::string model_filename = po.GetArg(3); // Output file
    std::string tree_filename = po.GetArg(4);  // Ouput file

    Vector<BaseFloat> glob_inv_var(dim);
    glob_inv_var.Set(1.0);
    Vector<BaseFloat> glob_mean(dim);
    glob_mean.Set(1.0);

    printf("----------- \n");
    /*
     * train_feats is an rspecifier for training features [used to set mean and variance]
     * In WSJ, this prints something like:
     * ark,s,cs:apply-cmvn  --utt2spk=ark:data/train_si84_2kshort/split10/1/utt2spk scp:data/train_si84_2kshort/split10/1/cmvn.scp scp:data/train_si84_2kshort/split10/1/feats.scp ark:- | add-deltas  ark:- ark:- | subset-feats --n=10 ark:- ark:-|
     **/
    printf(train_feats.c_str());
    printf("\n ---------------- \n");

    if (train_feats != "") {
      double count = 0.0;
      Vector<double> var_stats(dim);
      Vector<double> mean_stats(dim);
      SequentialDoubleMatrixReader feat_reader(train_feats);
      int32 num_feat_reads = 0;
      for (; !feat_reader.Done(); feat_reader.Next()) {
        num_feat_reads += 1;
        const Matrix<double> &mat = feat_reader.Value();
        std::cout << "Reading from matrix with " << std::to_string(mat.NumCols()) << " columns and " << std::to_string(mat.NumRows()) << " rows.\n";
        for (int32 i = 0; i < mat.NumRows(); i++) {
          count += 1.0;
          var_stats.AddVec2(1.0, mat.Row(i)); // Sum of squares for sample variance
          mean_stats.AddVec(1.0, mat.Row(i)); // Sum for sample mean
        }
      }
      std::cout << "Total num feat reads from the SequentialDoubleMatrixReader = " << std::to_string(num_feat_reads) << "\n";

      if (count == 0) { KALDI_ERR << "no features were seen."; }
      var_stats.Scale(1.0/count);
      mean_stats.Scale(1.0/count);
      var_stats.AddVec2(-1.0, mean_stats);
      if (var_stats.Min() <= 0.0)
        KALDI_ERR << "bad variance";
      var_stats.InvertElements();  // Why is this inverting elements?
      glob_inv_var.CopyFromVec(var_stats);
      glob_mean.CopyFromVec(mean_stats);
    }

    HmmTopology topo;
    bool binary_in;
    Input ki(topo_filename, &binary_in);
    topo.Read(ki.Stream(), binary_in); // It's reading from the XML-like file with the HMM topology (data/lang_nosp/topo)

    const std::vector<int32> &phones = topo.GetPhones();
    std::cout << "------------------------ \n";
    std::cout << "Phones from the HMM Topology:\n";
    for (auto i = phones.begin(); i != phones.end(); ++i) {
      std::cout << *i << ' ';
    }
    std::cout << "\n ------------------------ \n";

    std::vector<int32> phone2num_pdf_classes (1+phones.back());
    for (size_t i = 0; i < phones.size(); i++)
      phone2num_pdf_classes[phones[i]] = topo.NumPdfClasses(phones[i]);

    // Now the tree [not really a tree at this point]:
    ContextDependency *ctx_dep = NULL;
    if (shared_phones_rxfilename == "") {  // No sharing of phones: standard approach.
      ctx_dep = MonophoneContextDependency(phones, phone2num_pdf_classes);
    } else {
      std::vector<std::vector<int32> > shared_phones;
      ReadSharedPhonesList(shared_phones_rxfilename, &shared_phones);
      // ReadSharedPhonesList crashes on error.
      ctx_dep = MonophoneContextDependencyShared(shared_phones, phone2num_pdf_classes);
    }

    int32 num_pdfs = ctx_dep->NumPdfs();

    AmDiagGmm am_gmm;
    DiagGmm gmm;
    gmm.Resize(1, dim);
    {  // Initialize the gmm.
      Matrix<BaseFloat> inv_var(1, dim);
      inv_var.Row(0).CopyFromVec(glob_inv_var);
      Matrix<BaseFloat> mu(1, dim);
      mu.Row(0).CopyFromVec(glob_mean);
      Vector<BaseFloat> weights(1);
      weights.Set(1.0);
      gmm.SetInvVarsAndMeans(inv_var, mu);
      gmm.SetWeights(weights);
      gmm.ComputeGconsts();
    }

    for (int i = 0; i < num_pdfs; i++)
      am_gmm.AddPdf(gmm);

    if (perturb_factor != 0.0) {
      for (int i = 0; i < num_pdfs; i++)
        am_gmm.GetPdf(i).Perturb(perturb_factor);
    }

    // Now the transition model:
    TransitionModel trans_model(*ctx_dep, topo);

    {
      Output ko(model_filename, binary);
      trans_model.Write(ko.Stream(), binary);
      am_gmm.Write(ko.Stream(), binary);
    }

    // Now write the tree.
    ctx_dep->Write(Output(tree_filename, binary).Stream(),
                   binary);

    delete ctx_dep;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

