#!/usr/bin/env nextflow
/*
========================================================================================
                         nf-core/popbom
========================================================================================
 nf-core/popbom Analysis Pipeline.
 #### Homepage / Documentation
 https://github.com/nf-core/popbom
----------------------------------------------------------------------------------------
*/

def helpMessage() {
    // TODO nf-core: Add to this help message with new command line parameters
    log.info nfcoreHeader()
    log.info"""

    Usage:

    The typical command for running the pipeline is as follows:

    nextflow run nf-core/popbom --reads '*_R{1,2}.fastq.gz' -profile docker

    Mandatory arguments:
      --reads [file]                Path to input data (must be surrounded with quotes)
      -profile [str]                Configuration profile to use. Can use multiple (comma separated)
                                    Available: conda, docker, singularity, test, awsbatch, <institute> and more

    Options:
      --genome [str]                  Name of iGenomes reference
      --single_end [bool]             Specifies that the input is single-end reads


    Short read preprocessing:
      --adapter_forward             Sequence of 3' adapter to remove in the forward reads
      --adapter_reverse             Sequence of 3' adapter to remove in the reverse reads
      --mean_quality                Mean qualified quality value for keeping read (default: 15)
      --trimming_quality            Trimming quality value for the sliding window (default: 15)
      --keep_phix                   Keep reads similar to the Illumina internal standard PhiX genome (default: false)

    Taxonomy:
      --centrifuge_db [path]        Database for taxonomic binning with centrifuge (default: none). E.g. "ftp://ftp.ccb.jhu.edu/pub/infphilo/centrifuge/data/p_compressed+h+v.tar.gz"
      --kraken2_db [path]           Database for taxonomic binning with kraken2 (default: none). E.g. "ftp://ftp.ccb.jhu.edu/pub/data/kraken2_dbs/minikraken2_v2_8GB_201904_UPDATE.tgz"
      --metaphlan_db [path]         Database for taxonomic binning with metaphlan

    References                        If not specified in the configuration file or you wish to overwrite any of the references
      --fasta [file]                  Path to fasta reference

    Other options:
      --outdir [file]                 The output directory where the results will be saved
      --email [email]                 Set this parameter to your e-mail address to get a summary e-mail with details of the run sent to you when the workflow exits
      --email_on_fail [email]         Same as --email, except only send mail if the workflow is not successful
      --max_multiqc_email_size [str]  Theshold size for MultiQC report to be attached in notification email. If file generated by pipeline exceeds the threshold, it will not be attached (Default: 25MB)
      -name [str]                     Name for the pipeline run. If not specified, Nextflow will automatically generate a random mnemonic

    AWSBatch options:
      --awsqueue [str]                The AWSBatch JobQueue that needs to be set when running on AWSBatch
      --awsregion [str]               The AWS Region for your AWS Batch job to run on
      --awscli [str]                  Path to the AWS CLI tool
    """.stripIndent()
}

// Show help message
if (params.help) {
    helpMessage()
    exit 0
}

/*
 * SET UP CONFIGURATION VARIABLES
 */

/*
// Check if genome exists in the config file
if (params.genomes && params.genome && !params.genomes.containsKey(params.genome)) {
    exit 1, "The provided genome '${params.genome}' is not available in the iGenomes file. Currently the available genomes are ${params.genomes.keySet().join(", ")}"
}
*/


// Has the run name been specified by the user?
//  this has the bonus effect of catching both -name and --name
custom_runName = params.name
if (!(workflow.runName ==~ /[a-z]+_[a-z]+/)) {
    custom_runName = workflow.runName
}

if (workflow.profile.contains('awsbatch')) {
    // AWSBatch sanity checking
    if (!params.awsqueue || !params.awsregion) exit 1, "Specify correct --awsqueue and --awsregion parameters on AWSBatch!"
    // Check outdir paths to be S3 buckets if running on AWSBatch
    // related: https://github.com/nextflow-io/nextflow/issues/813
    if (!params.outdir.startsWith('s3:')) exit 1, "Outdir not on S3 - specify S3 Bucket to run on AWSBatch!"
    // Prevent trace files to be stored on S3 since S3 does not support rolling files.
    if (params.tracedir.startsWith('s3:')) exit 1, "Specify a local tracedir or run without trace! S3 cannot be used for tracefiles."
}


/*
 * short read preprocessing options
 */
params.adapter_forward = "AGATCGGAAGAGCACACGTCTGAACTCCAGTCA"
params.adapter_reverse = "AGATCGGAAGAGCGTCGTGTAGGGAAAGAGTGT"
params.mean_quality = 15
params.trimming_quality = 15
params.keep_phix = false
// params.phix_reference = "ftp://ftp.ncbi.nlm.nih.gov/genomes/genbank/viral/Enterobacteria_phage_phiX174_sensu_lato/all_assembly_versions/GCA_002596845.1_ASM259684v1/GCA_002596845.1_ASM259684v1_genomic.fna.gz"
params.phix_reference = "$baseDir/assets/data/GCA_002596845.1_ASM259684v1_genomic.fna.gz"


/*
 * taxonomy options
 */
params.skip_centrifuge = false
params.centrifuge_db = false
params.skip_kraken2 = false
params.kraken2_db = false
params.skip_metaphlan = false
params.metaphlan_db = false
params.metaphlan_read_min_len = 20



/*
 * ML options
 */
params.metadata = false
params.filter_species = true
params.filter_genus = false
params.combine_strain_species = false
params.seed = 42
params.loops_validation = 10
params.loops_tuning = 10
params.varience_threshold = false
params.scorer = 'balanced_accuracy'  //false
params.rf = false
params.svm = false
params.xgboost = false
params.l2linear = false


classifier_list = []
if(params.rf) { classifier_list.add('RF') } 
if(params.svm) { classifier_list.add('SVM') } 
if(params.xgboost) { classifier_list.add('XGB') } 
if(params.l2linear) { classifier_list.add('L2linear') } 


/*
 * Prediction options 
 */
params.predict = false
params.clf = false

/*
 * Check if parameters for host contamination removal are valid and create channels
 */
if ( params.host_fasta && params.host_genome) {
    exit 1, "Both host fasta reference and iGenomes genome are specififed to remove host contamination! Invalid combination, please specify either --host_fasta or --host_genome."
}
if ( params.manifest && (params.host_fasta || params.host_genome) ) {
    log.warn "Host read removal is only applied to short reads. Long reads might be filtered indirectly by Filtlong, which is set to use read qualities estimated based on k-mer matches to the short, already filtered reads."
    if ( params.longreads_length_weight > 1 ) {
        log.warn "The parameter --longreads_length_weight is ${params.longreads_length_weight}, causing the read length being more important for long read filtering than the read quality. Set --longreads_length_weight to 1 in order to assign equal weights."
    }
}

if ( params.host_genome ) {
    // Check if host genome exists in the config file
    if ( !params.genomes.containsKey(params.host_genome) ) {
        exit 1, "The provided host genome '${params.host_genome}' is not available in the iGenomes file. Currently the available genomes are ${params.genomes.keySet().join(", ")}"
    } else {
        host_fasta = params.genomes[params.host_genome].fasta ?: false
        if ( !host_fasta ) {
            exit 1, "No fasta file specified for the host genome ${params.host_genome}!"
        }
        Channel
            .value(file( "${host_fasta}", checkIfExists: true ))
            .set { ch_host_fasta }

        host_bowtie2index = params.genomes[params.host_genome].bowtie2 ?: false
        if ( !host_bowtie2index ) {
            exit 1, "No Bowtie 2 index file specified for the host genome ${params.host_genome}!"
        }
        Channel
            .value(file( "${host_bowtie2index}/*", checkIfExists: true ))
            .set { ch_host_bowtie2index }
    }
} else if ( params.host_fasta ) {
    Channel
        .value(file( "${params.host_fasta}", checkIfExists: true ))
        .set { ch_host_fasta }
} else {
    ch_host_fasta = Channel.empty()
}


// Stage config files
ch_multiqc_config = file("$baseDir/assets/multiqc_config.yaml", checkIfExists: true)
ch_multiqc_custom_config = params.multiqc_config ? Channel.fromPath(params.multiqc_config, checkIfExists: true) : Channel.empty()
ch_output_docs = file("$baseDir/docs/output.md", checkIfExists: true)

/*
 * Create a channel for input read files
 */

if(!params.keep_phix) {
    Channel
            .fromPath( "${params.phix_reference}", checkIfExists: true )
            .set { file_phix_db }
}

if(params.readPaths){
    if(params.singleEnd){
        Channel
                .from(params.readPaths)
                .map { row -> [ row[0], [file(row[1][0])]] }
                .ifEmpty { exit 1, "params.readPaths was empty - no input files supplied" }
                .into { read_files_fastqc; read_files_fastp }
        files_all_raw = Channel.from()
    } else {
        Channel
                .from(params.readPaths)
                .map { row -> [ row[0], [file(row[1][0]), file(row[1][1])]] }
                .ifEmpty { exit 1, "params.readPaths was empty - no input files supplied" }
                .into { read_files_fastqc; read_files_fastp }
        files_all_raw = Channel.from()
    }
} else {
    Channel
            .fromFilePairs( params.reads, size: params.singleEnd ? 1 : 2 )
            .ifEmpty { exit 1, "Cannot find any reads matching: ${params.reads}\nNB: Path needs to be enclosed in quotes!\nNB: Path requires at least one * wildcard!\nIf this is single-end data, please specify --singleEnd on the command line." }
            .into { read_files_fastqc; read_files_fastp }
    files_all_raw = Channel.from()
}


// Header log info
log.info nfcoreHeader()
def summary = [:]
if (workflow.revision) summary['Pipeline Release'] = workflow.revision
summary['Run Name']         = custom_runName ?: workflow.runName
// TODO nf-core: Report custom parameters here
summary['Reads']            = params.reads
summary['Fasta Ref']        = params.fasta
summary['Data Type']        = params.single_end ? 'Single-End' : 'Paired-End'
if (!params.skip_centrifuge) {
    if(params.centrifuge_db) summary['Centrifuge Db']         = params.centrifuge_db
} else {
    summary['Skip Centrifuge'] = 'Yes'
}
if (!params.skip_kraken2) {
    if(params.kraken2_db) summary['Kraken2 Db']         = params.kraken2_db
} else {
    summary['Skip kraken2'] = 'Yes'
}
if (!params.skip_metaphlan) {
    if(params.metaphlan_db) summary['Metaphlan Db']         = params.metaphlan_db
} else {
    summary['Skip Metaphlan'] = 'Yes'
}
summary['Max Resources']    = "$params.max_memory memory, $params.max_cpus cpus, $params.max_time time per job"
if (workflow.containerEngine) summary['Container'] = "$workflow.containerEngine - $workflow.container"
summary['Output dir']       = params.outdir
summary['Launch dir']       = workflow.launchDir
summary['Working dir']      = workflow.workDir
summary['Script dir']       = workflow.projectDir
summary['User']             = workflow.userName
if (workflow.profile.contains('awsbatch')) {
    summary['AWS Region']   = params.awsregion
    summary['AWS Queue']    = params.awsqueue
    summary['AWS CLI']      = params.awscli
}
summary['Config Profile'] = workflow.profile
if (params.config_profile_description) summary['Config Description'] = params.config_profile_description
if (params.config_profile_contact)     summary['Config Contact']     = params.config_profile_contact
if (params.config_profile_url)         summary['Config URL']         = params.config_profile_url
if (params.email || params.email_on_fail) {
    summary['E-mail Address']    = params.email
    summary['E-mail on failure'] = params.email_on_fail
    summary['MultiQC maxsize']   = params.max_multiqc_email_size
}
log.info summary.collect { k,v -> "${k.padRight(18)}: $v" }.join("\n")
log.info "-\033[2m--------------------------------------------------\033[0m-"

// Check the hostnames against configured profiles
checkHostname()

Channel.from(summary.collect{ [it.key, it.value] })
    .map { k,v -> "<dt>$k</dt><dd><samp>${v ?: '<span style=\"color:#999999;\">N/A</a>'}</samp></dd>" }
    .reduce { a, b -> return [a, b].join("\n            ") }
    .map { x -> """
    id: 'nf-core-popbom-summary'
    description: " - this information is collected when the pipeline is started."
    section_name: 'nf-core/popbom Workflow Summary'
    section_href: 'https://github.com/nf-core/popbom'
    plot_type: 'html'
    data: |
        <dl class=\"dl-horizontal\">
            $x
        </dl>
    """.stripIndent() }
    .set { ch_workflow_summary }

/*
 * Parse software version numbers
 */

process GET_METAPHLAN_VERSION {

    output:
    file "v_metaphlan.txt" into ch_metaphlan_version

    script:
    """
    metaphlan --version > v_metaphlan.txt
    """
}


process GET_SOFTWARE_VERSIONS {
    publishDir "${params.outdir}/pipeline_info", mode: 'copy',
        saveAs: { filename ->
                      if (filename.indexOf(".csv") > 0) filename
                      else null
                }
    input:
    file(metaphlan_version) from ch_metaphlan_version

    output:
    file 'software_versions_mqc.yaml' into ch_software_versions_yaml
    file "software_versions.csv"

    script:
    """
    echo $workflow.manifest.version > v_pipeline.txt
    echo $workflow.nextflow.version > v_nextflow.txt
    fastqc --version > v_fastqc.txt
    multiqc --version > v_multiqc.txt
    fastp -v 2> v_fastp.txt
    kraken2 -v > v_kraken2.txt
    centrifuge --version > v_centrifuge.txt
    #metaphlan -v > v_metaphlan.txt
    scrape_software_versions.py &> software_versions_mqc.yaml
    """
}


/*
 * STEP 1 - Read trimming and pre/post qc
 */
process FASTQC_RAW {
    tag "$name"
    publishDir "${params.outdir}/", mode: 'copy',
            saveAs: {filename -> filename.indexOf(".zip") == -1 ? "QC_shortreads/fastqc/$filename" : null}

    input:
    set val(name), file(reads) from read_files_fastqc

    output:
    file "*_fastqc.{zip,html}" into fastqc_results

    script:
    """
    fastqc -t "${task.cpus}" -q $reads
    """
}


process FASTP {
    tag "$name"
    publishDir "${params.outdir}/", mode: 'copy',
            saveAs: {filename -> filename.indexOf(".fastq.gz") == -1 ? "QC_shortreads/fastp/$name/$filename" : null}

    input:
    set val(name), file(reads) from read_files_fastp
    val adapter from params.adapter_forward
    val adapter_reverse from params.adapter_reverse
    val qual from params.mean_quality
    val trim_qual from params.trimming_quality

    output:
    set val(name), file("${name}_trimmed*.fastq.gz") into trimmed_reads
    file("fastp.*")

    script:
    if ( params.singleEnd ) {
        """
            fastp -w "${task.cpus}" -q "${qual}" --cut_by_quality5 \
            --cut_by_quality3 --cut_mean_quality "${trim_qual}"\
            --adapter_sequence=${adapter} --adapter_sequence_r2=${adapter_reverse} \
            -i "${reads[0]}" \
            --stdout | pigz -p \"${task.cpus}\" --best > \"${name}_trimmed.fastq.gz\"
            """
    } else {
        """
            fastp -w "${task.cpus}" -q "${qual}" --cut_by_quality5 \
            --cut_by_quality3 --cut_mean_quality "${trim_qual}"\
            --adapter_sequence=${adapter} --adapter_sequence_r2=${adapter_reverse} \
            -i "${reads[0]}" -I "${reads[1]}" \
            --stdout | \
            paste - - - - - - - - | tee >( \
            cut -f 1-4 | tr "\t" "\n" | egrep -v '^\$' | \
            pigz -p \"${task.cpus}\" --best  > \"${name}_trimmed_R1.fastq.gz\"\
            ) | \
            cut -f 5-8 | tr "\t" "\n" | egrep -v '^\$' | \
            pigz -p \"${task.cpus}\" --best  > \"${name}_trimmed_R2.fastq.gz\"
            """
    }
}


/*
 * Remove host read contamination
 */
(trimmed_reads, ch_trimmed_reads_remove_host) = trimmed_reads.into(2)

process HOST_BOWTIE2INDEX {
    tag "${genome}"

    input:
    file(genome) from ch_host_fasta

    output:
    file("bt2_index_base*") into ch_host_bowtie2index

    when: params.host_fasta

    script:
    """
    bowtie2-build --threads "${task.cpus}" "${genome}" "bt2_index_base"
    """
}

process REMOVE_HOST {
    tag "${name}"

    publishDir "${params.outdir}/QC_shortreads/remove_host/", mode: params.publish_dir_mode,
        saveAs: {filename ->
                    if (filename.indexOf(".fastq.gz") == -1) "$filename"
                    else null
                }

    input:
    set val(name), file(reads) from ch_trimmed_reads_remove_host
    file(index) from ch_host_bowtie2index

    output:
    set val(name), file("${name}_host_unmapped*.fastq.gz") into ch_trimmed_reads_host_removed
    file("${name}.bowtie2.log") into ch_host_removed_log
    file("${name}_host_mapped*.read_ids.txt") optional true

    when: params.host_fasta || params.host_genome

    script:
    def sensitivity = params.host_removal_verysensitive ? "--very-sensitive" : "--sensitive"
    def save_ids = params.host_removal_save_ids ? "Y" : "N"
    if ( !params.single_end ) {
        """
        bowtie2 -p "${task.cpus}" \
                -x ${index[0].getSimpleName()} \
                -1 "${reads[0]}" -2 "${reads[1]}" \
                $sensitivity \
                --un-conc-gz ${name}_host_unmapped_%.fastq.gz \
                --al-conc-gz ${name}_host_mapped_%.fastq.gz \
                1> /dev/null \
                2> ${name}.bowtie2.log
        if [ ${save_ids} = "Y" ] ; then
            zcat ${name}_host_mapped_1.fastq.gz | awk '{if(NR%4==1) print substr(\$0, 2)}' | LC_ALL=C sort > ${name}_host_mapped_1.read_ids.txt
            zcat ${name}_host_mapped_2.fastq.gz | awk '{if(NR%4==1) print substr(\$0, 2)}' | LC_ALL=C sort > ${name}_host_mapped_2.read_ids.txt
        fi
        rm -f ${name}_host_mapped_*.fastq.gz
        """
    } else {
        """
        bowtie2 -p "${task.cpus}" \
                -x ${index[0].getSimpleName()} \
                -U ${reads} \
                $sensitivity \
                --un-gz ${name}_host_unmapped.fastq.gz \
                --al-gz ${name}_host_mapped.fastq.gz \
                1> /dev/null \
                2> ${name}.bowtie2.log
        if [ ${save_ids} = "Y" ] ; then
            zcat ${name}_host_mapped.fastq.gz | awk '{if(NR%4==1) print substr(\$0, 2)}' | LC_ALL=C sort > ${name}_host_mapped.read_ids.txt
        fi
        rm -f ${name}_host_mapped.fastq.gz
        """
    }
}

if ( params.host_fasta || params.host_genome ) trimmed_reads = ch_trimmed_reads_host_removed
else ch_trimmed_reads_remove_host.close()



/*
 * Remove PhiX contamination from Illumina reads
 */
if(!params.keep_phix) {
    process PHIX_DOWNLOAD_DB {
        tag "${genome}"

        input:
        file(genome) from file_phix_db

        output:
        set file(genome), file("ref*") into phix_db

        script:
        """
        bowtie2-build --threads "${task.cpus}" "${genome}" ref
        """
    }

    process REMOVE_PHIX {
        tag "$name"

        publishDir "${params.outdir}", mode: 'copy',
                saveAs: {filename -> filename.indexOf(".fastq.gz") == -1 ? "QC_shortreads/remove_phix/$filename" : null}

        input:
        set val(name), file(reads), file(genome), file(db) from trimmed_reads.combine(phix_db)

        output:
        set val(name), file("*.fastq.gz") into (trimmed_reads_fastqc, trimmed_reads_centrifuge, trimmed_reads_kraken2, trimmed_reads_metaphlan)
        file("${name}_remove_phix_log.txt")

        script:
        if ( !params.singleEnd ) {
            """
            bowtie2 -p "${task.cpus}" -x ref -1 "${reads[0]}" -2 "${reads[1]}" --un-conc-gz ${name}_unmapped_%.fastq.gz
            echo "Bowtie2 reference: ${genome}" >${name}_remove_phix_log.txt
            zcat ${reads[0]} | echo "Read pairs before removal: \$((`wc -l`/4))" >>${name}_remove_phix_log.txt
            zcat ${name}_unmapped_1.fastq.gz | echo "Read pairs after removal: \$((`wc -l`/4))" >>${name}_remove_phix_log.txt
            """
        } else {
            """
            bowtie2 -p "${task.cpus}" -x ref -U ${reads}  --un-gz ${name}_unmapped.fastq.gz
            echo "Bowtie2 reference: ${genome}" >${name}_remove_phix_log.txt
            zcat ${reads[0]} | echo "Reads before removal: \$((`wc -l`/4))" >>${name}_remove_phix_log.txt
            zcat ${name}_unmapped.fastq.gz | echo "Reads after removal: \$((`wc -l`/4))" >>${name}_remove_phix_log.txt
            """
        }

    }
} else {
    trimmed_reads.into {trimmed_reads_fastqc; trimmed_reads_centrifuge; trimmed_reads_kraken2; trimmed_reads_metaphlan}
}


process FASTQC_TRIMMED {
    tag "$name"
    publishDir "${params.outdir}/", mode: 'copy',
            saveAs: {filename -> filename.indexOf(".zip") == -1 ? "QC_shortreads/fastqc/$filename" : null}

    input:
    set val(name), file(reads) from trimmed_reads_fastqc

    output:
    file "*_fastqc.{zip,html}" into fastqc_results_trimmed

    script:
    if ( !params.single_end ) {
        """
        fastqc -t "${task.cpus}" -q ${reads}
        mv *1_fastqc.html "${name}_R1.trimmed_fastqc.html"
        mv *2_fastqc.html "${name}_R2.trimmed_fastqc.html"
        mv *1_fastqc.zip "${name}_R1.trimmed_fastqc.zip"
        mv *2_fastqc.zip "${name}_R2.trimmed_fastqc.zip"
        """
    } else {
        """
        fastqc -t "${task.cpus}" -q ${reads}
        mv *_fastqc.html "${name}.trimmed_fastqc.html"
        mv *_fastqc.zip "${name}.trimmed_fastqc.zip"
        """
    }
}

// TODO rewrite channel io
if ( !params.skip_centrifuge ) {
        process CENTRIFUGE_DB_PREPARATION {
            input:
            file(db) from centrifuge_db

            output:
            set val("${db.toString().replace(".tar.gz", "")}"), file("*.cf") into centrifuge_database

            script:
            """
            tar -xf "${db}"
            """
    }


    trimmed_reads_centrifuge
            .combine(centrifuge_database)
            .set { centrifuge_input }

    process CENTRIFUGE {
        tag "${name}-${db_name}"
        publishDir "${params.outdir}/Taxonomy/centrifuge/${name}", mode: 'copy'

        input:
        set val(name), file(reads), val(db_name), file(db) from centrifuge_input

        output:
        set val("centrifuge"), val(name), file("results.krona") into centrifuge_to_krona
        file("report.txt")
        file("kreport.txt")

        script:
        def input = params.singleEnd ? "-U \"${reads}\"" :  "-1 \"${reads[0]}\" -2 \"${reads[1]}\""
        """
        centrifuge -x "${db_name}" \
            -p "${task.cpus}" \
            --report-file report.txt \
            -S results.txt \
            $input
        centrifuge-kreport -x "${db_name}" results.txt > kreport.txt
        """
    }

}

// PREPROCESSING: uncompressing kraken2 db

if ( !params.skip_kraken2) {
	if ( params.kraken2_db ) {
    file(params.kraken2_db, checkIfExists: true)
	    if (params.kraken2_db.endsWith('.tar.gz') || params.kraken2_db.endsWith('.tgz')) {

        process UNTAR_KRAKEN2_DB {
            label 'error_retry'
            if (params.save_reference) {
                publishDir "${params.outdir}/genome", mode: params.publish_dir_mode
            }

            input:
            path db from params.kraken2_db

            output:
            file("$dbname*") into ch_kraken2_db

            script:
	    dbname = params.kraken2_db.tokenize("/")[-1].tokenize(".")[0]
            """
            tar -xvzf $db
            """
        }
    } else {
        ch_kraken2_db = file(params.kraken2_db)
    }
}

// PREPROCESSING: Build Kraken2 db

    if ( !params.kraken2_db ) {

        process KRAKEN2_BUILD {
            tag "$db"
            label 'process_high'
            if (params.save_reference) {
                publishDir "${params.outdir}/genome", mode: params.publish_dir_mode
            }

            output:
            path "$db" into ch_kraken2_db

            script:
            db = "kraken2_db"
            ftp = params.kraken2_use_ftp ? "--use-ftp" : ""
            """
            kraken2-build --standard --db $db --threads $task.cpus $ftp
            """
        }
    }

    process KRAKEN2 {
        tag "${name}"
        publishDir "${params.outdir}/Taxonomy/kraken2/${name}", mode: 'copy'

        input:
        tuple val(name), file(reads) from trimmed_reads_kraken2
        file(kraken2_db) from ch_kraken2_db

        output:
        file("*.txt") into ch_kraken_reports

        script:
        def input = params.singleEnd ? "\"${reads}\"" :  "--paired \"${reads[0]}\" \"${reads[1]}\""
        """
        kraken2 \
            --report-zero-counts \
            --threads "${task.cpus}" \
            --db "${kraken2_db}"  \
            --report ${name}.txt \
            $input
        """
    }
} else { ch_kraken_reports = Channel.from() } 


if ( !params.skip_metaphlan) {
	if ( !params.metaphlan_db ) {

    process METAPHLAN_DB_PREPARATION {
        
        output:
        path "$db" into (ch_metaphlan_db, ch_metaphlan_db_strain)

        script:
        db = "metaphlan_db"
        """
        metaphlan --install --bowtie2db "${db}"
        """
    }
} else {
    ch_metaphlan_db = Channel.fromPath(params.metaphlan_db)
    ch_metaphlan_db_strain = Channel.fromPath(params.metaphlan_db)
}

    trimmed_reads_metaphlan
            .combine(ch_metaphlan_db)
            .set { ch_metaphlan_input }


    process METAPHLAN {
        tag "${name}"
        publishDir "${params.outdir}/Taxonomy/metaphlan/${name}", mode: 'copy'

        input:
        tuple val(name), file(reads), file(db) from ch_metaphlan_input
        val metaphlan_read_min_len from params.metaphlan_read_min_len

        output:
        file("*.txt") into ch_metaphlan_report
        tuple val(name), file("mapping.bt2") into ch_metaphlan_strain

        script:
        def input = params.singleEnd ? "\"${reads}\"" :  "\"${reads[0]}\",\"${reads[1]}\""
        """
        metaphlan \
            $input \
            --input_type fastq \
            --bowtie2db "${db}" \
            --bowtie2out mapping.bt2 \
            --nproc "${task.cpus}" \
            --read_min_len "${metaphlan_read_min_len}" \
            -o ${name}.mpa.txt
        """
    }

    ch_metaphlan_strain.combine(ch_metaphlan_db_strain).set { ch_metaphlan_strain_input }

    process STRAIN {
    tag "${name}"
        publishDir "${params.outdir}/Taxonomy/metaphlan_strain/${name}", mode: 'copy'

        input:
        tuple val(name), file(mapping), file(db) from ch_metaphlan_strain_input 

        output:
        file("*.txt") into ch_metaphlan_strain_report

        """
        metaphlan \
            -t marker_pres_table \
            "${mapping}" \
            --input_type bowtie2out \
            --bowtie2db "${db}" \
            --nproc "${task.cpus}" \
            -o ${name}.marker.txt
        """
    }
}

process TAXO_REPORT_SUMMARY {
    publishDir "${params.outdir}/summary_tables", mode: 'copy'
 
    input:
    path meta from params.metadata
    file mpa from ch_metaphlan_report.collect().ifEmpty([])
    file mpa_marker from ch_metaphlan_strain_report.collect().ifEmpty([])
    file kraken from ch_kraken_reports.collect().ifEmpty([])

    output:
    path "*.csv" into ch_reports_summary

    script:
    def metaphlan = params.skip_metaphlan ? "" : "--metaphlan \"$mpa\" --mpa_marker \"$mpa_marker\""
    def kraken2 = params.skip_kraken2 ? "" : "--kraken2 \"$kraken\""
    def species = params.filter_species ? "--filter_level species" : ""
    def genus = params.filter_genus ? "--filter_level genus" : ""
    def combine = params.combine_strain_species ? "--combine" : ""
    def metadata = params.metadata ? "--metadata $meta" : ""
    """
    create_summary_tables.py \
        $metadata \
        $species \
        $metaphlan \
        $combine \
        $kraken2
    """
}

if(!params.predict) {
    process MODEL_TRAINING {
        tag "${filename}-${classifier}"
        publishDir "${params.outdir}/classification_metrics/${classifier}", mode: 'copy'
        publishDir "${params.outdir}/classifier", pattern: "*.joblib"
        
        input:
        path report from ch_reports_summary.buffer( size: 1 ).flatten()
        val seed from params.seed
        val lv from params.loops_validation
        val lt from params.loops_tuning
        val scorer from params.scorer
        each classifier from classifier_list
        val vt from params.varience_threshold

        output:
        file("*")

        script:
        filename = report.toString().replace("_table.csv", "").tokenize('/')[-1]
        varience = params.varience_threshold ? "--varience_threshold " + params.varience_threshold : ''
        """
        train.py \
        --input $report \
        --seed $seed \
        --threads "${task.cpus}" \
        --loops_validation $lv \
        --loops_tuning $lt \
        --classifier $classifier \
        --scorer $scorer \
        $varience \
        --output ${filename}_${classifier}
        """
    }
} else {
    ch_clf = Channel.fromPath(params.clf, checkIfExists: true).map{file -> [ file.name.tokenize("_")[0], file ]}
    ch_reports = ch_reports_summary.buffer( size: 1 ).flatten().map{file -> [ file.name.tokenize("_")[0], file ]}
    ch_clf.join(ch_reports)
// table and clf have to fit together!!!

    process PREDICTION {
        publishDir "${params.outdir}/prediction", mode: 'copy'

        input:
        tuple val(name), path(clf), path(report) classifier from ch_clf

        output:
        file("*_prediction.tsv")

        script:
        """
        predict.py \
        --input $report\
        --output $name\
        --classifier $clf
        """
    }
}


/*
 * MultiQC
 */

process MULTIQC {
    publishDir "${params.outdir}/MultiQC", mode: 'copy'

    input:
    file (multiqc_config) from ch_multiqc_config
    file (mqc_custom_config) from ch_multiqc_custom_config.collect().ifEmpty([])
    // TODO nf-core: Add in log files from your new processes for MultiQC to find!
    file (fastqc_raw:'fastqc/*') from fastqc_results.collect().ifEmpty([])
    file (fastqc_trimmed:'fastqc/*') from fastqc_results_trimmed.collect().ifEmpty([])
    file (host_removal) from ch_host_removed_log.collect().ifEmpty([])
    file ('software_versions/*') from ch_software_versions_yaml.collect()
    file workflow_summary from ch_workflow_summary.collectFile(name: "workflow_summary_mqc.yaml")

    output:
    file "*multiqc_report.html" into ch_multiqc_report
    file "*_data"
    file "multiqc_plots"

    script:
    rtitle = custom_runName ? "--title \"$custom_runName\"" : ''
    rfilename = custom_runName ? "--filename " + custom_runName.replaceAll('\\W','_').replaceAll('_+','_') + "_multiqc_report" : ''
    custom_config_file = params.multiqc_config ? "--config $mqc_custom_config" : ''
    read_type = params.single_end ? "--single_end" : ''
    if ( params.host_fasta || params.host_genome ) {
        """
        # get multiqc parsed data for bowtie 2
        multiqc -f $rtitle $rfilename $custom_config_file *.bowtie2.log
        multiqc_to_custom_tsv.py ${read_type}
        # run multiqc using custom content file instead of original bowtie2 log files
        multiqc -f $rtitle $rfilename $custom_config_file --ignore "*.bowtie2.log" .
        """
    } else {
        """
        multiqc -f $rtitle $rfilename $custom_config_file .
        """
    }
}


/*
 * Output Description HTML
 */
process OUTPUT_DOCUMENTATION {
    publishDir "${params.outdir}/pipeline_info", mode: 'copy'

    input:
    file output_docs from ch_output_docs

    output:
    file "results_description.html"

    script:
    """
    markdown_to_html.py $output_docs -o results_description.html
    """
}

/*
 * Completion e-mail notification
 */
workflow.onComplete {

    // Set up the e-mail variables
    def subject = "[nf-core/popbom] Successful: $workflow.runName"
    if (!workflow.success) {
        subject = "[nf-core/popbom] FAILED: $workflow.runName"
    }
    def email_fields = [:]
    email_fields['version'] = workflow.manifest.version
    email_fields['runName'] = custom_runName ?: workflow.runName
    email_fields['success'] = workflow.success
    email_fields['dateComplete'] = workflow.complete
    email_fields['duration'] = workflow.duration
    email_fields['exitStatus'] = workflow.exitStatus
    email_fields['errorMessage'] = (workflow.errorMessage ?: 'None')
    email_fields['errorReport'] = (workflow.errorReport ?: 'None')
    email_fields['commandLine'] = workflow.commandLine
    email_fields['projectDir'] = workflow.projectDir
    email_fields['summary'] = summary
    email_fields['summary']['Date Started'] = workflow.start
    email_fields['summary']['Date Completed'] = workflow.complete
    email_fields['summary']['Pipeline script file path'] = workflow.scriptFile
    email_fields['summary']['Pipeline script hash ID'] = workflow.scriptId
    if (workflow.repository) email_fields['summary']['Pipeline repository Git URL'] = workflow.repository
    if (workflow.commitId) email_fields['summary']['Pipeline repository Git Commit'] = workflow.commitId
    if (workflow.revision) email_fields['summary']['Pipeline Git branch/tag'] = workflow.revision
    email_fields['summary']['Nextflow Version'] = workflow.nextflow.version
    email_fields['summary']['Nextflow Build'] = workflow.nextflow.build
    email_fields['summary']['Nextflow Compile Timestamp'] = workflow.nextflow.timestamp


    // On success try attach the multiqc report
    def mqc_report = null
    try {
        if (workflow.success) {
            mqc_report = ch_multiqc_report.getVal()
            if (mqc_report.getClass() == ArrayList) {
                log.warn "[nf-core/popbom] Found multiple reports from process 'multiqc', will use only one"
                mqc_report = mqc_report[0]
            }
        }
    } catch (all) {
        log.warn "[nf-core/popbom] Could not attach MultiQC report to summary email"
    }

    // Check if we are only sending emails on failure
    email_address = params.email
    if (!params.email && params.email_on_fail && !workflow.success) {
        email_address = params.email_on_fail
    }

    // Render the TXT template
    def engine = new groovy.text.GStringTemplateEngine()
    def tf = new File("$baseDir/assets/email_template.txt")
    def txt_template = engine.createTemplate(tf).make(email_fields)
    def email_txt = txt_template.toString()

    // Render the HTML template
    def hf = new File("$baseDir/assets/email_template.html")
    def html_template = engine.createTemplate(hf).make(email_fields)
    def email_html = html_template.toString()

    // Render the sendmail template
    def smail_fields = [ email: email_address, subject: subject, email_txt: email_txt, email_html: email_html, baseDir: "$baseDir", mqcFile: mqc_report, mqcMaxSize: params.max_multiqc_email_size.toBytes() ]
    def sf = new File("$baseDir/assets/sendmail_template.txt")
    def sendmail_template = engine.createTemplate(sf).make(smail_fields)
    def sendmail_html = sendmail_template.toString()

    // Send the HTML e-mail
    if (email_address) {
        try {
            if (params.plaintext_email) { throw GroovyException('Send plaintext e-mail, not HTML') }
            // Try to send HTML e-mail using sendmail
            [ 'sendmail', '-t' ].execute() << sendmail_html
            log.info "[nf-core/popbom] Sent summary e-mail to $email_address (sendmail)"
        } catch (all) {
            // Catch failures and try with plaintext
            [ 'mail', '-s', subject, email_address ].execute() << email_txt
            log.info "[nf-core/popbom] Sent summary e-mail to $email_address (mail)"
        }
    }

    // Write summary e-mail HTML to a file
    def output_d = new File("${params.outdir}/pipeline_info/")
    if (!output_d.exists()) {
        output_d.mkdirs()
    }
    def output_hf = new File(output_d, "pipeline_report.html")
    output_hf.withWriter { w -> w << email_html }
    def output_tf = new File(output_d, "pipeline_report.txt")
    output_tf.withWriter { w -> w << email_txt }

    c_green = params.monochrome_logs ? '' : "\033[0;32m";
    c_purple = params.monochrome_logs ? '' : "\033[0;35m";
    c_red = params.monochrome_logs ? '' : "\033[0;31m";
    c_reset = params.monochrome_logs ? '' : "\033[0m";

    if (workflow.stats.ignoredCount > 0 && workflow.success) {
        log.info "-${c_purple}Warning, pipeline completed, but with errored process(es) ${c_reset}-"
        log.info "-${c_red}Number of ignored errored process(es) : ${workflow.stats.ignoredCount} ${c_reset}-"
        log.info "-${c_green}Number of successfully ran process(es) : ${workflow.stats.succeedCount} ${c_reset}-"
    }

    if (workflow.success) {
        log.info "-${c_purple}[nf-core/popbom]${c_green} Pipeline completed successfully${c_reset}-"
    } else {
        checkHostname()
        log.info "-${c_purple}[nf-core/popbom]${c_red} Pipeline completed with errors${c_reset}-"
    }

}


def nfcoreHeader() {
    // Log colors ANSI codes
    c_black = params.monochrome_logs ? '' : "\033[0;30m";
    c_blue = params.monochrome_logs ? '' : "\033[0;34m";
    c_cyan = params.monochrome_logs ? '' : "\033[0;36m";
    c_dim = params.monochrome_logs ? '' : "\033[2m";
    c_green = params.monochrome_logs ? '' : "\033[0;32m";
    c_purple = params.monochrome_logs ? '' : "\033[0;35m";
    c_reset = params.monochrome_logs ? '' : "\033[0m";
    c_white = params.monochrome_logs ? '' : "\033[0;37m";
    c_yellow = params.monochrome_logs ? '' : "\033[0;33m";

    return """    -${c_dim}--------------------------------------------------${c_reset}-
                                            ${c_green},--.${c_black}/${c_green},-.${c_reset}
    ${c_blue}        ___     __   __   __   ___     ${c_green}/,-._.--~\'${c_reset}
    ${c_blue}  |\\ | |__  __ /  ` /  \\ |__) |__         ${c_yellow}}  {${c_reset}
    ${c_blue}  | \\| |       \\__, \\__/ |  \\ |___     ${c_green}\\`-._,-`-,${c_reset}
                                            ${c_green}`._,._,\'${c_reset}
    ${c_purple}  nf-core/popbom v${workflow.manifest.version}${c_reset}
    -${c_dim}--------------------------------------------------${c_reset}-
    """.stripIndent()
}

def checkHostname() {
    def c_reset = params.monochrome_logs ? '' : "\033[0m"
    def c_white = params.monochrome_logs ? '' : "\033[0;37m"
    def c_red = params.monochrome_logs ? '' : "\033[1;91m"
    def c_yellow_bold = params.monochrome_logs ? '' : "\033[1;93m"
    if (params.hostnames) {
        def hostname = "hostname".execute().text.trim()
        params.hostnames.each { prof, hnames ->
            hnames.each { hname ->
                if (hostname.contains(hname) && !workflow.profile.contains(prof)) {
                    log.error "====================================================\n" +
                            "  ${c_red}WARNING!${c_reset} You are running with `-profile $workflow.profile`\n" +
                            "  but your machine hostname is ${c_white}'$hostname'${c_reset}\n" +
                            "  ${c_yellow_bold}It's highly recommended that you use `-profile $prof${c_reset}`\n" +
                            "============================================================"
                }
            }
        }
    }
}
